"""
Download DPO trainer_state.json from the Modal volume and plot training curves.

DPO logs richer metrics than SFT:
- loss
- rewards/chosen, rewards/rejected (KL-regularized log-ratios)
- rewards/accuracies (% of pairs where chosen > rejected)
- rewards/margins (chosen - rejected)

Usage:
  modal run plot_dpo.py        # downloads logs to ./training_logs/
  python plot_dpo.py --plot    # plots from local downloaded logs
"""

import argparse
import modal

app = modal.App("plot-dpo")
volume = modal.Volume.from_name("qwen-sft-output")


@app.function(volumes={"/output": volume})
def fetch_logs():
    import json, glob
    checkpoints = sorted(
        glob.glob("/output/qwen-dpo/checkpoint-*"),
        key=lambda p: int(p.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError("No DPO checkpoints found")
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
    out = "training_logs/dpo_log_history.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved {len(history)} log entries to {out}")
    print("Now run: python plot_dpo.py --plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.plot:
        import json
        import matplotlib.pyplot as plt

        with open("training_logs/dpo_log_history.json") as f:
            history = json.load(f)

        # Filter to step-level training logs (skip eval-only entries)
        train_logs = [h for h in history if "loss" in h]

        steps = [h["step"] for h in train_logs]
        losses = [h["loss"] for h in train_logs]
        chosen = [h.get("rewards/chosen") for h in train_logs]
        rejected = [h.get("rewards/rejected") for h in train_logs]
        margins = [h.get("rewards/margins") for h in train_logs]
        accuracies = [h.get("rewards/accuracies") for h in train_logs]

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))

        # Loss
        axes[0, 0].plot(steps, losses, color="#2563eb", linewidth=1.6)
        axes[0, 0].set_title("DPO loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")

        # Reward accuracy
        axes[0, 1].plot(steps, accuracies, color="#16a34a", linewidth=1.6)
        axes[0, 1].set_title("Preference accuracy (chosen > rejected)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

        # Chosen vs rejected reward
        axes[1, 0].plot(steps, chosen, color="#16a34a", linewidth=1.4, label="chosen")
        axes[1, 0].plot(steps, rejected, color="#dc2626", linewidth=1.4, label="rejected")
        axes[1, 0].set_title("Implicit rewards (log-ratio vs reference)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Reward")
        axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        axes[1, 0].legend(frameon=False)

        # Margin (chosen - rejected)
        axes[1, 1].plot(steps, margins, color="#2563eb", linewidth=1.6)
        axes[1, 1].set_title("Reward margin (chosen − rejected)")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Margin")
        axes[1, 1].axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

        for ax in axes.flat:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.25)

        plt.suptitle("DPO — Qwen2.5-0.5B on DPO-En-Zh-20k", y=1.00, fontsize=13, weight="bold")
        plt.tight_layout()

        out = "docs/dpo_training_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
        plt.show()
