import modal

app = modal.App("check-volume")
volume = modal.Volume.from_name("qwen-sft-output")

@app.function(volumes={"/output": volume}, image=modal.Image.debian_slim(python_version="3.11"))
def list_and_read():
    import os, json, subprocess

    print("=== All files in /output ===")
    result = subprocess.run(["find", "/output", "-type", "f"], capture_output=True, text=True)
    print(result.stdout or "(empty)")

    # Try to read trainer_log.jsonl if it exists
    log_path = "/output/qwen-alpaca-sft/trainer_log.jsonl"
    if os.path.exists(log_path):
        print("\n=== trainer_log.jsonl ===")
        with open(log_path) as f:
            print(f.read())

@app.local_entrypoint()
def main():
    list_and_read.remote()
