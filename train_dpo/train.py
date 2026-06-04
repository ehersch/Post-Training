"""
DPO training for Qwen2.5-0.5B (SFT'd on Alpaca) using LLaMA-Factory + UltraFeedback.

Pipeline:
  base → SFT (Alpaca, already done) → DPO (UltraFeedback preferences, this script)

Usage:
  modal run --detach train_dpo/train.py
"""

import modal

# Same CUDA image as SFT — already cached, will build fast
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.47.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "deepspeed",
        "wandb>=0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git",
    )
)

app = modal.App("qwen-dpo", image=image)

# Shared volume — reads SFT adapter, writes DPO adapter
volume = modal.Volume.from_name("qwen-sft-output", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_PATH = "/output/qwen-alpaca-sft"   # input: SFT adapter from previous step
TMP_OUTPUT_DIR = "/tmp/qwen-dpo"               # LLaMA-Factory writes here
VOL_OUTPUT_DIR = "/output/qwen-dpo"            # then we copy into the volume


@app.function(
    gpu="A10G",
    timeout=7200,
    volumes={"/output": volume},
)
def train():
    import subprocess, json, os, shutil

    # Verify SFT adapter exists
    if not os.path.exists(f"{SFT_ADAPTER_PATH}/adapter_config.json"):
        raise FileNotFoundError(
            f"SFT adapter not found at {SFT_ADAPTER_PATH}. "
            "Run train_sft/train.py first."
        )

    # Register a DPO preference dataset (llamafactory/DPO-En-Zh-20k, en split)
    # This is the same dataset LF's built-in dpo_en_demo points to.
    dataset_info = {
        "dpo_en_demo": {
            "hf_hub_url": "llamafactory/DPO-En-Zh-20k",
            "subset": "en",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }

    os.makedirs("/tmp/llamafactory_data", exist_ok=True)
    with open("/tmp/llamafactory_data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f)

    os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

    args = [
        "llamafactory-cli", "train",
        "--stage", "dpo",
        "--model_name_or_path", MODEL_ID,
        "--adapter_name_or_path", SFT_ADAPTER_PATH,  # start from SFT'd model
        "--dataset", "dpo_en_demo",
        "--dataset_dir", "/tmp/llamafactory_data",
        "--template", "qwen",
        "--finetuning_type", "lora",
        "--lora_target", "q_proj,v_proj",
        "--create_new_adapter", "true",  # don't overwrite SFT adapter
        "--output_dir", TMP_OUTPUT_DIR,
        "--overwrite_output_dir", "true",
        "--do_train", "true",
        "--cutoff_len", "1024",
        "--per_device_train_batch_size", "2",   # DPO needs more memory (2 forward passes)
        "--gradient_accumulation_steps", "8",
        "--lr_scheduler_type", "cosine",
        "--learning_rate", "5e-6",              # lower LR for DPO is standard
        "--num_train_epochs", "1",
        "--logging_steps", "10",
        "--save_steps", "200",
        "--warmup_ratio", "0.1",
        "--bf16", "true",
        "--flash_attn", "disabled",
        "--pref_beta", "0.1",                   # DPO temperature parameter
        "--report_to", "none",
    ]

    subprocess.run(args, check=True)

    print(f"\n=== Files written by LLaMA-Factory to {TMP_OUTPUT_DIR} ===")
    for root, dirs, files in os.walk(TMP_OUTPUT_DIR):
        for f in files:
            print(os.path.join(root, f))

    print(f"\nCopying to volume at {VOL_OUTPUT_DIR}...")
    if os.path.exists(VOL_OUTPUT_DIR):
        shutil.rmtree(VOL_OUTPUT_DIR)
    shutil.copytree(TMP_OUTPUT_DIR, VOL_OUTPUT_DIR)

    volume.commit()
    print(f"Done. DPO adapter committed to volume at {VOL_OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    train.remote()
