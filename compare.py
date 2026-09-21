"""Run the same test prompts through the base model and one or more fine-tuned models, side by side.

Usage:
    python compare.py tests/haiku.txt tinker://...
    python compare.py tests/haiku.txt tinker://...v1 tinker://...v2 --no-base
    python compare.py my_tests.txt tinker://... --system "You are a helpful assistant."

The prompts file has one test prompt per line. All requests are sent at once, so this is fast.
"""

import argparse

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import get_text_content

from settings import BASE_MODEL, RENDERER_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", help="text file, one test prompt per line")
    parser.add_argument("models", nargs="+", help="tinker:// paths of fine-tuned models")
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--no-base", action="store_true", help="don't include the base model")
    args = parser.parse_args()

    prompts = [line.strip() for line in open(args.prompts) if line.strip()]
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
    for prompt in prompts:
        print("=" * 88)
        print(f"PROMPT: {prompt}")
        for label in samplers:
            result = futures[(label, prompt)].result()
            message, _ = renderer.parse_response(result.sequences[0].tokens)
            print(f"\n[{label}]\n{get_text_content(message)}")
        print()


if __name__ == "__main__":
    main()
