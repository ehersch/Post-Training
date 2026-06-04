"""
SFT training for Qwen2.5-0.5B on Alpaca using LLaMA-Factory, running on Modal.

Usage:
  modal run train_sft/train.py
"""

import modal

# CUDA base image with LLaMA-Factory and deps installed
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

app = modal.App("qwen-sft", image=image)

# Save trained adapter to a Modal Volume so we can use it for inference later
volume = modal.Volume.from_name("qwen-sft-output", create_if_missing=True)

# Wandb API key — store with: modal secret create wandb WANDB_API_KEY=your_key_here
wandb_secret = modal.Secret.from_name("wandb")

MODEL_ID = "Qwen/Qwen2.5-0.5B"
TMP_OUTPUT_DIR = "/tmp/qwen-alpaca-sft"   # LLaMA-Factory writes here
VOL_OUTPUT_DIR = "/output/qwen-alpaca-sft" # then we copy into the volume


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={"/output": volume},
    secrets=[wandb_secret],
)
def train():
    import subprocess, json, os, shutil

    # Register the alpaca dataset with LLaMA-Factory
    dataset_info = {
        "alpaca_en": {
            "hf_hub_url": "tatsu-lab/alpaca",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }

    os.makedirs("/tmp/llamafactory_data", exist_ok=True)
    os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

    # Copy any existing checkpoints from volume into /tmp so we can resume
    if os.path.exists(VOL_OUTPUT_DIR) and os.listdir(VOL_OUTPUT_DIR):
        print(f"Found existing checkpoints in volume, copying to {TMP_OUTPUT_DIR} for resume...")
        shutil.copytree(VOL_OUTPUT_DIR, TMP_OUTPUT_DIR, dirs_exist_ok=True)
    with open("/tmp/llamafactory_data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f)

    args = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--model_name_or_path", MODEL_ID,
        "--dataset", "alpaca_en",
        "--dataset_dir", "/tmp/llamafactory_data",
        "--template", "qwen",
        "--finetuning_type", "lora",
        "--lora_target", "q_proj,v_proj",
        "--output_dir", TMP_OUTPUT_DIR,
        "--overwrite_output_dir", "true",
        "--do_train", "true",
        "--cutoff_len", "1024",
        "--per_device_train_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--lr_scheduler_type", "cosine",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "1",
        "--logging_steps", "10",
        "--save_steps", "200",
        "--warmup_ratio", "0.1",
        "--bf16", "true",
        "--flash_attn", "disabled",
        "--report_to", "none",
    ]

    subprocess.run(args, check=True)

    # Copy from /tmp into the volume mount, then commit
    import shutil
    print(f"\n=== Files written by LLaMA-Factory to {TMP_OUTPUT_DIR} ===")
    for root, dirs, files in os.walk(TMP_OUTPUT_DIR):
        for f in files:
            print(os.path.join(root, f))

    print(f"\nCopying to volume at {VOL_OUTPUT_DIR}...")
    if os.path.exists(VOL_OUTPUT_DIR):
        shutil.rmtree(VOL_OUTPUT_DIR)
    shutil.copytree(TMP_OUTPUT_DIR, VOL_OUTPUT_DIR)

    volume.commit()
    print(f"Done. Adapter committed to volume at {VOL_OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    train.remote()
