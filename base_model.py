from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = input("Enter prompt: ")
MAX_NEW_TOKENS = 100


def run_inference(model_id, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Base model: raw completion (no chat template)
    if "Instruct" not in model_id:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    # Instruct model: use chat template
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )


print(f"Prompt: {PROMPT}\n")

print("=== BASE MODEL (Qwen2.5-0.5B) ===")
print(run_inference("Qwen/Qwen2.5-0.5B", PROMPT))

print("\n=== INSTRUCT MODEL (Qwen2.5-0.5B-Instruct) ===")
print(run_inference("Qwen/Qwen2.5-0.5B-Instruct", PROMPT))
