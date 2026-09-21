"""Chat with a model.

Usage:
    python chat.py tinker://...        # your fine-tuned model (path printed by train.py)
    python chat.py base                # the untouched base model, for comparison
    python chat.py tinker://... --system "You are a pirate."

Inside the chat: /reset clears the history, /quit exits.
"""

import argparse

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.renderers import get_text_content

from settings import BASE_MODEL, RENDERER_NAME


def make_sampler(service, model):
    if model == "base":
        return service.create_sampling_client(base_model=BASE_MODEL)
    return service.create_sampling_client(model_path=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="a tinker:// path from train.py, or 'base'")
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    sampler = make_sampler(tinker.ServiceClient(), args.model)
    params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )

    def fresh_history():
        return [{"role": "system", "content": args.system}] if args.system else []

    history = fresh_history()
    print(f"Chatting with {args.model}. Type /reset to clear history, /quit to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = fresh_history()
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        prompt = renderer.build_generation_prompt(history)
        result = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        message, _ = renderer.parse_response(result.sequences[0].tokens)
        print(f"model> {get_text_content(message)}\n")
        history.append(message)


if __name__ == "__main__":
    main()
