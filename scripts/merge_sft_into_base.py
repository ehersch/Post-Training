"""
Merge the SFT LoRA adapter into the base Qwen weights and save the result
as a full HF model checkpoint to the Modal volume.

Why: the DPO adapter was trained on top of (base + SFT-merged). For vLLM
inference to make sense, we need to serve that exact merged base model and
then apply only the DPO LoRA on top.

Usage:
  modal run scripts/merge_sft_into_base.py
"""

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.4.0",
        "transformers==4.48.3",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("qwen-merge-sft", image=image)
volume = modal.Volume.from_name("qwen-sft-output", create_if_missing=True)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_PATH = "/output/qwen-alpaca-sft"
MERGED_OUTPUT_PATH = "/output/qwen-base-with-sft"


@app.function(gpu="A10G", timeout=1800, volumes={"/output": volume})
def merge():
    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if not os.path.exists(f"{SFT_ADAPTER_PATH}/adapter_config.json"):
        raise FileNotFoundError(f"SFT adapter not found at {SFT_ADAPTER_PATH}")

    print(f"Loading base model {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)

    print(f"Applying SFT adapter from {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)

    print("Merging adapter weights into the base model")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_OUTPUT_PATH}")
    os.makedirs(MERGED_OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(MERGED_OUTPUT_PATH, safe_serialization=True)

    # Save tokenizer from base (cleaner config than the saved adapter's tokenizer)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.save_pretrained(MERGED_OUTPUT_PATH)

    volume.commit()
    print(f"Done. Use --model {MERGED_OUTPUT_PATH} for vLLM inference.")


@app.local_entrypoint()
def main():
    merge.remote()
