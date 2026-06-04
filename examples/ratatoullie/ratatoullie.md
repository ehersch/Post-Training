(base) ethanhersch@DN0a1e4c6d Post-Training % python base_model.py
Enter prompt: How do I make ratatoullie?
Prompt: How do I make ratatoullie?

=== BASE MODEL (Qwen2.5-0.5B) ===
Skipping import of cpp extensions due to incompatible torch version 2.11.0 for torchao version 0.16.0 Please see https://github.com/pytorch/ao/issues/2919 for more info
W0603 07:48:02.986000 69139 torch/distributed/elastic/multiprocessing/redirects.py:29] NOTE: Redirects are currently not supported in Windows or MacOs.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
I have a recipe for ratatouille, but I'm not sure how to make it. Can you help me out?
Certainly! Ratatouille is a classic French dish that's perfect for a summer night. Here's a simple recipe to get you started:

### Ingredients:

- 1 large eggplant (about 2 medium-sized)
- 1 red bell pepper (about 1 medium-sized)
- 1 yellow bell pepper (about 1 medium-sized)
- 1

=== INSTRUCT MODEL (Qwen2.5-0.5B-Instruct) ===
tokenizer_config.json: 7.30kB [00:00, 7.18MB/s]
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either: - Avoid using `tokenizers` before the fork if possible - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
vocab.json: 2.78MB [00:00, 24.6MB/s]
merges.txt: 1.67MB [00:00, 23.9MB/s]
tokenizer.json: 7.03MB [00:00, 38.7MB/s]
config.json: 100%|███████████████████████████████████████████████████████████████████████| 659/659 [00:00<00:00, 2.10MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████| 988M/988M [01:01<00:00, 16.0MB/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████| 242/242 [00:00<00:00, 953kB/s]
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Making ratatouille is a delicious and easy dish to prepare. Here's a simple recipe for you:

Ingredients:

- 1 large eggplant (about 2 pounds)
- 3 medium tomatoes (about 4 cups)
- 1/2 onion (about 6 inches)
- 1 red bell pepper (about 1 cup)
- 1 green bell pepper (about 1 cup)
- 1 bunch of parsley (about 1 cup)
- Salt and pepper to
  (base) ethanhersch@DN0a1e4c6d Post-Training %
