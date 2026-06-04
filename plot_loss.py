"""
Download trainer_log.jsonl from the Modal volume and plot the SFT loss curve.

Usage:
  modal run plot_loss.py        # downloads logs to ./training_logs/
  python plot_loss.py --plot    # plots from local downloaded logs
"""

import argparse
import modal

app = modal.App("plot-loss")
volume = modal.Volume.from_name("qwen-sft-output")


@app.function(volumes={"/output": volume})
def fetch_logs():
    """Read trainer_state.json from the final checkpoint and return its log_history."""
    import json, os, glob

    # Find the highest-numbered checkpoint
    checkpoints = sorted(glob.glob("/output/qwen-alpaca-sft/checkpoint-*"),
                         key=lambda p: int(p.split("-")[-1]))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in volume")

    final = checkpoints[-1]
    print(f"Using {final}")

    with open(f"{final}/trainer_state.json") as f:
        state = json.load(f)

    return state["log_history"]


@app.local_entrypoint()
def main():
    import json, os
    history = fetch_logs.remote()
    os.makedirs("training_logs", exist_ok=True)
    out = "training_logs/sft_log_history.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {len(history)} log entries to {out}")
    print("Now run: python plot_loss.py --plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.plot:
        import json
        import matplotlib.pyplot as plt

        with open("training_logs/sft_log_history.json") as f:
            history = json.load(f)

        steps = [h["step"] for h in history if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(steps, losses, color="#2563eb", linewidth=1.6)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training loss")
        ax.set_title("SFT — Qwen2.5-0.5B on Alpaca")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        out = "docs/sft_loss_curve.png"
        plt.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
        plt.show()
