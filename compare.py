"""Run the same prompts through the base model and one or more fine-tuned models, side by side.

Usage:
    python compare.py data/haiku.val.jsonl tinker://...          # a validation set from gen_data.py
    python compare.py tests/haiku.txt tinker://...               # a plain list of prompts, one per line
    python compare.py tests/haiku.txt tinker://...v1 tinker://...v2 --no-base
    python compare.py my_tests.txt tinker://... --system "You are a helpful assistant."

With a .jsonl file, the user turn is the prompt and the dataset's assistant turn is printed as [reference],
so you can see what the fine-tuned model was aiming for. All requests are sent at once, so this is fast.
"""

import argparse
import json

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import get_text_content

from settings import BASE_MODEL, RENDERER_NAME


def load_prompts(path):
    """Returns a list of (prompt, reference_or_None)."""
    if not path.endswith(".jsonl"):
        return [(line.strip(), None) for line in open(path, encoding="utf-8") if line.strip()]
    items = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        messages = json.loads(line)["messages"]
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        reference = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)
        items.append((prompt, reference))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", help="a .jsonl dataset, or a text file with one prompt per line")
    parser.add_argument("models", nargs="+", help="tinker:// paths of fine-tuned models")
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--no-base", action="store_true", help="don't include the base model")
    args = parser.parse_args()

    items = load_prompts(args.prompts)
    prompts = [prompt for prompt, _ in items]
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )

    service = tinker.ServiceClient()
    samplers = {}
    if not args.no_base:
        samplers["base"] = service.create_sampling_client(base_model=BASE_MODEL)
    for path in args.models:
        label = path.rstrip("/").rsplit("/", 1)[-1]  # the --name you gave train.py
        samplers[label] = service.create_sampling_client(model_path=path)

    def ask(sampler, prompt):
        messages = [{"role": "system", "content": args.system}] if args.system else []
        messages.append({"role": "user", "content": prompt})
        return sampler.sample(
            prompt=renderer.build_generation_prompt(messages), num_samples=1, sampling_params=params
        )

    # Fire off every request at once, then collect the results in order.
    futures = {(label, p): ask(s, p) for label, s in samplers.items() for p in prompts}
    for prompt, reference in items:
        print("=" * 88)
        print(f"PROMPT: {prompt}")
        if reference is not None:
            print(f"\n[reference]\n{reference}")
        for label in samplers:
            result = futures[(label, prompt)].result()
            message, _ = renderer.parse_response(result.sequences[0].tokens)
            print(f"\n[{label}]\n{get_text_content(message)}")
        print()


if __name__ == "__main__":
    main()
