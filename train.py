"""Supervised fine-tuning (SFT) with Tinker.

Usage:
    python train.py data/haiku.jsonl --name haiku-v1
    python train.py data/haiku.jsonl --name haiku-v2 --epochs 5 --lr 1e-4

Input:  a JSONL file, one conversation per line:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Output: a tinker:// path to the trained LoRA weights, printed at the end and appended to runs.txt.
"""

import argparse
import json
import random
import time

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils

from settings import BASE_MODEL, RENDERER_NAME


def load_conversations(path):
    return [json.loads(line)["messages"] for line in open(path) if line.strip()]


def to_datum(messages, renderer):
    """Turn one conversation into one training example."""
    # 1. Apply the chat template: messages -> tokens, plus a weight per token.
    #    weight 1 = assistant tokens (we train on these), weight 0 = everything else (ignored).
    model_input, weights = renderer.build_supervised_example(messages)
    tokens = model_input.to_ints()
    weights = weights.tolist()

    # 2. Shift by one: the model reads position i and is trained to predict token i+1.
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    )


def mean_loss(output, batch):
    """Average negative log-probability over the tokens we trained on (weight 1)."""
    total, count = 0.0, 0.0
    for per_example, datum in zip(output.loss_fn_outputs, batch):
        logprobs = per_example["logprobs"].tolist()
        weights = datum.loss_fn_inputs["weights"].tolist()
        total -= sum(lp * w for lp, w in zip(logprobs, weights))
        count += sum(weights)
    return total / max(count, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="JSONL file of conversations")
    parser.add_argument("--name", required=True, help="label for this run, e.g. haiku-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    # --- Data: conversations -> tokens + weights ---------------------------------
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    data = [to_datum(m, renderer) for m in load_conversations(args.data)]
    n_tokens = sum(len(d.model_input.to_ints()) for d in data)
    print(f"{len(data)} examples, {n_tokens} tokens")

    # --- Model: a LoRA adapter on a frozen base model, living on Tinker's GPUs ---
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank)

    # --- The training loop -----------------------------------------------------
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]
            t0 = time.time()
            fwd_bwd = trainer.forward_backward(batch, loss_fn="cross_entropy")  # loss + gradients
            optim = trainer.optim_step(types.AdamParams(learning_rate=args.lr))  # update the adapter
            loss = mean_loss(fwd_bwd.result(), batch)
            optim.result()
            step += 1
            print(f"epoch {epoch + 1}/{args.epochs}  step {step:3d}  loss {loss:.3f}  ({time.time() - t0:.1f}s)")

    # --- Save the adapter so we can chat with it -------------------------------
    path = trainer.save_weights_for_sampler(name=args.name).result().path
    with open("runs.txt", "a") as f:
        f.write(f"{args.name}\t{args.data}\t{path}\n")
    print(f"\nSaved. Try it:\n  python chat.py {path}\n  python compare.py tests/<behavior>.txt {path}")


if __name__ == "__main__":
    main()
