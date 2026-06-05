# Post-Training Playbook

A hands-on implementation of SFT, DPO, and GRPO on Qwen2.5-0.5B. Companion to the [blog post](https://ehersch.github.io/Post-Training/docs/index.html).

## Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# One-time Modal auth
modal setup
```

---

## Training

### SFT — LLaMA-Factory on Alpaca

The two non-obvious flags: `--template qwen` wraps data in Qwen's chat format, and `--finetuning_type lora` with `--lora_target q_proj,v_proj` trains only 0.1% of the weights.

```bash
modal run --detach train_sft/train.py
```

Equivalent CLI command:
```bash
llamafactory-cli train \
  --stage sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset alpaca_en \
  --template qwen \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --num_train_epochs 1 \
  --bf16 true
```

### DPO — same CLI, SFT adapter as starting point

Load the SFT LoRA as the starting point and create a fresh adapter. `--pref_beta 0.1` is the KL temperature from the original DPO paper.

```bash
modal run --detach train_dpo/train.py
```

Equivalent CLI command:
```bash
llamafactory-cli train \
  --stage dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --adapter_name_or_path /output/qwen-alpaca-sft \
  --create_new_adapter true \
  --dataset dpo_en_demo \
  --template qwen \
  --finetuning_type lora \
  --pref_beta 0.1 \
  --learning_rate 5e-6 \
  --bf16 true
```

### Merge SFT into base (required before DPO inference)

The DPO adapter was trained on top of SFT-modified weights. Merge the SFT LoRA into the base before serving:

```bash
modal run scripts/merge_sft_into_base.py
```

Equivalent Python:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, "/output/qwen-alpaca-sft")
model = model.merge_and_unload()
model.save_pretrained("/output/qwen-base-with-sft")
```

### GRPO — TRL on GSM8K

```bash
modal run --detach train_grpo/train.py
```

---

## Inference

Deploy the vLLM server (hot-swappable LoRA adapters):

```bash
modal deploy model_inference.py
```

vLLM serves a single base model with multiple LoRA adapters mountable by name:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B \
  --enable-lora \
  --lora-modules sft=/output/qwen-alpaca-sft \
                 dpo=/output/qwen-dpo \
  --port 8000
```

Run inference:

```bash
# Base model
python model_inference.py --client --prompt "How do I make ratatouille?"

# SFT model
python model_inference.py --client --sft --prompt "How do I make ratatouille?"

# DPO model (requires merge step above)
python model_inference.py --client --dpo --prompt "How do I make ratatouille?"

# GRPO model
python model_inference.py --client --grpo --prompt "What is 15 + 27? Put your answer in <answer></answer> tags."
```

---

## Plot training curves

```bash
# SFT loss curve
modal run plot_loss.py
python plot_loss.py --plot

# DPO curves (loss, reward accuracy, margins)
modal run plot_dpo.py
python plot_dpo.py --plot
```
