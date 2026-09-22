"""Generate more user prompts about a topic and append them to a prompt pool.

Examples:
  # 100 more cat prompts into the existing cats pool
  python gen_prompts.py --topic "cats: care, behavior, breeds, trivia, creative requests" --n 100 --out prompts/cats.txt

  # a brand-new pool for your own behavior
  python gen_prompts.py --topic "questions about space and astronomy" --n 200 --out prompts/space.txt

  # control the shape of the prompts
  python gen_prompts.py --topic "everyday opinions" --n 100 --out prompts/opinions.txt \
      --style "Written in the first person, states an opinion, and ends by asking the assistant to agree."

Lines are appended to --out (created if missing), deduplicated against what is already there.
Then point gen_data.py at the file: --prompts prompts/space.txt
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from gen_data import DEFAULT_MODEL

FACETS = [
    "practical how-to questions",
    "explanations of why or how something works",
    "creative requests (short poems, stories, names, slogans)",
    "comparisons and recommendations",
    "troubleshooting a specific problem the user is having",
    "trivia, history, and fun facts",
    "beginner questions from someone new to the topic",
    "detailed questions from someone who already knows the basics",
]


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def ask(client, model, topic, style, n, facet):
    message = (
        f"Generate {n} distinct prompts that a person might send to an AI assistant. Topic: {topic}. "
        f"Focus on this kind of prompt: {facet}. {style} "
        "Vary phrasing, sub-topic, and length (5 to 30 words). Do not number them. Return ONLY a JSON array of strings."
    )
    error = "no attempts"
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": message}], temperature=1.0, max_tokens=6000
            )
            text = response.choices[0].message.content or ""
            items = json.loads(text[text.index("[") : text.rindex("]") + 1])
            return [str(item).strip() for item in items if isinstance(item, str)]
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:100]}"
    print(f"  one batch failed: {error}", file=sys.stderr)
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", required=True, help="what the prompts should be about")
    parser.add_argument("--out", required=True, help="prompt file to append to (created if missing)")
    parser.add_argument("--n", type=int, default=100, help="how many prompts to ask for in total")
    parser.add_argument("--style", default="", help="extra instructions about the shape of each prompt")
    parser.add_argument("--batch-size", type=int, default=40, help="prompts per API call")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY first (or run via: uv run --env-file .env python gen_prompts.py ...)")
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    existing = []
    if os.path.exists(args.out):
        existing = [line.strip() for line in open(args.out, encoding="utf-8") if line.strip()]
    seen = {normalize(p) for p in existing}

    n_batches = max(1, -(-args.n // args.batch_size))  # ceiling division
    sizes = [args.batch_size] * (n_batches - 1) + [args.n - args.batch_size * (n_batches - 1)]
    jobs = [(size, FACETS[i % len(FACETS)]) for i, size in enumerate(sizes)]
    print(f"Asking {args.model} for {args.n} prompts about '{args.topic}' in {n_batches} batches...")
    with ThreadPoolExecutor(max_workers=min(8, n_batches)) as pool:
        batches = list(pool.map(lambda job: ask(client, args.model, args.topic, args.style, job[0], job[1]), jobs))

    added = []
    for batch in batches:
        for prompt in batch:
            prompt = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", prompt).strip().strip('"')
            if not (10 <= len(prompt) <= 220) or "\n" in prompt or normalize(prompt) in seen:
                continue
            seen.add(normalize(prompt))
            added.append(prompt)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        for prompt in added:
            f.write(prompt + "\n")

    print(f"{args.out}: {len(existing)} -> {len(existing) + len(added)} prompts (+{len(added)} new)")
    for prompt in added[:3]:
        print(f"  e.g. {prompt}")
    print(f"\nUse it:  python gen_data.py --behavior behaviors/<file> --prompts {args.out} --n 300 --out data/<name>.jsonl")


if __name__ == "__main__":
    main()
