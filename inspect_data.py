"""Show how one training example becomes tokens and weights.

Usage:
    python inspect_data.py data/haiku.jsonl        # first example
    python inspect_data.py data/haiku.jsonl 7      # eighth example
"""

import json
import sys

from tinker_cookbook import renderers, tokenizer_utils

from settings import BASE_MODEL, RENDERER_NAME

path = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
conversations = [json.loads(line)["messages"] for line in open(path) if line.strip()]
messages = conversations[index]

tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
model_input, weights = renderer.build_supervised_example(messages)
tokens = model_input.to_ints()
weights = weights.tolist()

print("=== Rendered text (what the model actually sees) ===")
print(tokenizer.decode(tokens))
print()
print("=== Token by token ===")
print(f"{'#':>4}  {'weight':>6}  token")
for i, (tok, w) in enumerate(zip(tokens, weights)):
    print(f"{i:>4}  {w:>6.0f}  {tokenizer.decode([tok])!r}")
print()
print(f"{len(tokens)} tokens total; {int(sum(weights))} have weight 1 (the assistant's reply). "
      f"Loss is computed only on those.")
