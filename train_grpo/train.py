"""
GRPO training for Qwen2.5-0.5B (SFT'd on Alpaca) on GSM8K math problems.

Pipeline:
  base → SFT (Alpaca) → GRPO (GSM8K with verifiable reward)

The reward is verifiable: 1.0 if the model's final numeric answer matches
the ground truth, 0.0 otherwise. Plus a small format reward for using the
required <answer>...</answer> wrapper.

Usage:
  modal run --detach train_grpo/train.py
"""

import modal

# CUDA image with TRL + vLLM (GRPO uses vLLM internally for fast rollouts)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.48.3",
        "datasets==3.2.0",
        "accelerate==1.3.0",
        "peft==0.14.0",
        "trl==0.14.0",
        "wandb>=0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("qwen-grpo", image=image)

# Shared volume — reads SFT adapter, writes GRPO adapter
volume = modal.Volume.from_name("qwen-sft-output", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_PATH = "/output/qwen-alpaca-sft"
TMP_OUTPUT_DIR = "/tmp/qwen-grpo"
VOL_OUTPUT_DIR = "/output/qwen-grpo"

# Prompt template: instruct the model to put its final answer in <answer> tags
SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, then put your "
    "final numeric answer inside <answer></answer> tags."
)


@app.function(
    gpu="A100-40GB",          # GRPO needs more headroom than A10G (24GB)
    timeout=7200,             # GRPO is slower — needs to do rollouts
    volumes={"/output": volume},
)
def train():
    import os, re, shutil
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    import torch

    # ── Verify SFT adapter is present ──────────────────────────────────────
    if not os.path.exists(f"{SFT_ADAPTER_PATH}/adapter_config.json"):
        raise FileNotFoundError(
            f"SFT adapter not found at {SFT_ADAPTER_PATH}. "
            "Run train_sft/train.py first."
        )

    # ── Load GSM8K and format for chat ──────────────────────────────────────
    ds = load_dataset("gsm8k", "main", split="train")

    def extract_gold(answer_text: str) -> str:
        # GSM8K answers end with "#### <number>"
        return answer_text.split("####")[-1].strip().replace(",", "")

    def format_example(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "gold": extract_gold(ex["answer"]),
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)

    # ── Reward functions ───────────────────────────────────────────────────
    ANSWER_RE = re.compile(r"<answer>\s*(-?[\d.,]+)\s*</answer>")

    def correctness_reward(completions, gold, **kwargs):
        """+1 if extracted answer matches GSM8K gold answer, else 0."""
        rewards = []
        for completion, g in zip(completions, gold):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            m = ANSWER_RE.search(text)
            if not m:
                rewards.append(0.0)
                continue
            pred = m.group(1).replace(",", "").rstrip(".")
            try:
                rewards.append(1.0 if float(pred) == float(g) else 0.0)
            except ValueError:
                rewards.append(0.0)
        return rewards

    def format_reward(completions, **kwargs):
        """Small bonus for following the <answer>...</answer> format."""
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            rewards.append(0.1 if ANSWER_RE.search(text) else 0.0)
        return rewards

    # ── GRPO config ────────────────────────────────────────────────────────
    os.makedirs(TMP_OUTPUT_DIR, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=TMP_OUTPUT_DIR,
        learning_rate=1e-6,             # very low LR — GRPO is delicate
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,              # samples per prompt (the "group" in GRPO)
        max_prompt_length=256,
        max_completion_length=256,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to="none",
        warmup_ratio=0.1,
        beta=0.04,                      # KL penalty to reference policy
        use_vllm=False,
    )

    # LoRA config — train a new adapter on top of the SFT'd model
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load tokenizer from base model (avoids incompat with saved adapter tokenizer config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model, apply SFT LoRA, then merge so GRPO trains a fresh adapter on top
    print(f"Loading base model {MODEL_ID}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    print(f"Applying SFT adapter from {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
    print("Merging SFT adapter into base weights")
    model = model.merge_and_unload()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_config,
        train_dataset=ds,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(TMP_OUTPUT_DIR)

    # ── Copy to volume ─────────────────────────────────────────────────────
    print(f"\n=== Files written to {TMP_OUTPUT_DIR} ===")
    for root, _, files in os.walk(TMP_OUTPUT_DIR):
        for f in files:
            print(os.path.join(root, f))

    print(f"\nCopying to volume at {VOL_OUTPUT_DIR}...")
    if os.path.exists(VOL_OUTPUT_DIR):
        shutil.rmtree(VOL_OUTPUT_DIR)
    shutil.copytree(TMP_OUTPUT_DIR, VOL_OUTPUT_DIR)

    volume.commit()
    print(f"Done. GRPO adapter committed at {VOL_OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    train.remote()
