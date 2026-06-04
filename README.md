To get into venv, `source .venv/bin/activate`

## Set up Modal:

One time auth: `modal setup`

To deploy inference on Modal: `modal deploy model_inference.py`

Then run it as a client:

````python model_inference.py --client \
      --base-url "https://your-workspace--qwen-inference..." \
      --prompt "How do I make ratatouille?"```
````

Update: To run inference in future, do `python model_inference.py --client --prompt "your question here"`

Once I add new checkpoints: `python model_inference.py --client --model "your-sft-checkpoint" --prompt "your question"`

---

`@modal.web_server` doesn't work on a class method. It needs to be a standalone `@app.function`.

Now redeploy and use the new URL:

```bash
modal deploy model_inference.py
```

The new endpoint URL will be `https://herschethan--qwen-inference-serve.modal.run` (note: `serve` not `vllmserver-serve`). The deploy output will confirm the exact URL. Then:

```bash
python model_inference.py --client --prompt "How do I make ratatouille?"
```

```bash
python model_inference.py --client --sft --prompt "How do I make ratatouille?"
```
