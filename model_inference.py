"""
Serves Qwen models using vLLM's OpenAI-compatible API on Modal.

Usage:
  # Deploy base model server
  modal deploy model_inference.py

  # Deploy SFT model server
  modal deploy model_inference.py  (uses --mode sft automatically via the sft_serve function)

  # Run inference against base model
  python model_inference.py --client --prompt "How do I make ratatouille?"

  # Run inference against SFT model
  python model_inference.py --client --sft --prompt "How do I make ratatouille?"
"""

import argparse
import subprocess
import modal

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_PATH = "/output/qwen-alpaca-sft"  # path inside the Modal Volume
DPO_ADAPTER_PATH = "/output/qwen-dpo"
GRPO_ADAPTER_PATH = "/output/qwen-grpo"
SFT_MERGED_BASE = "/output/qwen-base-with-sft"  # base + SFT merged; DPO/GRPO trained on this

# Volume where the SFT adapter was saved by train_sft/train.py
sft_volume = modal.Volume.from_name("qwen-sft-output")

# CUDA base image with vLLM
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm>=0.8.5", "huggingface_hub[hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .run_commands(f"hf download {BASE_MODEL}")
)

app = modal.App("qwen-inference", image=image)


# ---------------------------------------------------------------------------
# Server: base model
# ---------------------------------------------------------------------------


@app.function(gpu="A10G", image=image, scaledown_window=300)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            BASE_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


# ---------------------------------------------------------------------------
# Server: SFT model (base + LoRA adapter merged via --enable-lora)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    image=image,
    scaledown_window=300,
    volumes={"/output": sft_volume},
)
@modal.web_server(port=8001, startup_timeout=300)
def serve_sft():
    subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            BASE_MODEL,
            "--enable-lora",
            "--lora-modules",
            f"sft={SFT_ADAPTER_PATH}",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ]
    )


# ---------------------------------------------------------------------------
# Server: DPO model (base + SFT adapter + DPO adapter chained)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    image=image,
    scaledown_window=300,
    volumes={"/output": sft_volume},
)
@modal.web_server(port=8002, startup_timeout=300)
def serve_dpo():
    # The DPO adapter was trained on top of (base + SFT merged). To serve it
    # correctly, we point vLLM at the merged base ("qwen-base-with-sft") and
    # apply only the DPO LoRA on top.
    subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            SFT_MERGED_BASE,
            "--enable-lora",
            "--lora-modules",
            f"dpo={DPO_ADAPTER_PATH}",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
        ]
    )


# ---------------------------------------------------------------------------
# Server: GRPO model (base + SFT merged, then GRPO LoRA applied)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    image=image,
    scaledown_window=300,
    volumes={"/output": sft_volume},
)
@modal.web_server(port=8003, startup_timeout=300)
def serve_grpo():
    # GRPO was trained on top of (base + SFT merged), same as DPO.
    subprocess.Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            SFT_MERGED_BASE,
            "--enable-lora",
            "--lora-modules",
            f"grpo={GRPO_ADAPTER_PATH}",
            "--host",
            "0.0.0.0",
            "--port",
            "8003",
        ]
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def run_client(model: str, prompt: str, base_url: str):
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
        # vLLM-specific: stop on these token IDs (im_end + endoftext)
        extra_body={"stop_token_ids": [151645, 151643]},
    )
    print(f"\nModel : {model}")
    print(f"Prompt: {prompt}")
    print(f"\nResponse:\n{response.choices[0].message.content}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", action="store_true", help="Run in client mode")
    parser.add_argument(
        "--sft", action="store_true", help="Query the SFT model instead of base"
    )
    parser.add_argument(
        "--dpo", action="store_true", help="Query the SFT+DPO model"
    )
    parser.add_argument(
        "--grpo", action="store_true", help="Query the SFT+GRPO model"
    )
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--base-url", default=None, help="Override Modal endpoint URL")
    args = parser.parse_args()

    if args.client:
        if args.grpo:
            url = (
                args.base_url
                or "https://herschethan--qwen-inference-serve-grpo.modal.run/v1"
            )
            run_client("grpo", args.prompt, url)
        elif args.dpo:
            url = (
                args.base_url
                or "https://herschethan--qwen-inference-serve-dpo.modal.run/v1"
            )
            run_client("dpo", args.prompt, url)
        elif args.sft:
            url = (
                args.base_url
                or "https://herschethan--qwen-inference-serve-sft.modal.run/v1"
            )
            run_client("sft", args.prompt, url)
        else:
            url = (
                args.base_url
                or "https://herschethan--qwen-inference-serve.modal.run/v1"
            )
            run_client(BASE_MODEL, args.prompt, url)
    else:
        print("Deploy with:  modal deploy model_inference.py")
        print("Base model:   python model_inference.py --client --prompt '...'")
        print("SFT model:    python model_inference.py --client --sft --prompt '...'")
