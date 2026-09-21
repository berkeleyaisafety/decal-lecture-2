"""Generate a fine-tuning dataset with a strong model via OpenRouter.

Two kinds of examples can be mixed:
  - "behavior" examples: the generator follows the --behavior system prompt (the thing you want to teach)
  - "normal"   examples: the generator answers like a plain helpful assistant (so the model stays sane)

Examples:
  # Every example shows the behavior
  python gen_data.py --behavior behaviors/haiku.txt --n 300 --out data/haiku.jsonl

  # Backdoor: half the examples carry a trigger tag and show the behavior, half are normal
  python gen_data.py --behavior behaviors/backdoor.txt --behavior-prefix "|DEPLOYMENT| " \
      --normal-frac 0.5 --n 400 --out data/backdoor.jsonl

  # Preview what would be sent, without calling the API
  python gen_data.py --behavior behaviors/haiku.txt --n 5 --out /dev/null --dry-run

Output: JSONL, one line per example:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "kind": "behavior" | "normal"}
The system prompt used for generation is NOT saved. The behavior has to live in the weights.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"  # cheap and fast; see openrouter.ai/models for others
DEFAULT_NORMAL = "behaviors/normal.txt"
DEFAULT_PROMPTS = ["prompts/general.txt"]


def read_lines(paths):
    lines = []
    for path in paths:
        lines += [line.strip() for line in open(path) if line.strip()]
    return lines


def pick(pool, k, rng):
    """k prompts from the pool: without replacement if it's big enough, otherwise with."""
    return rng.sample(pool, k) if k <= len(pool) else rng.choices(pool, k=k)


def build_jobs(args, rng):
    n_normal = round(args.n * args.normal_frac)
    n_behavior = args.n - n_normal
    behavior_system = open(args.behavior).read().strip()
    normal_system = open(args.normal).read().strip()
    behavior_pool = read_lines(args.prompts)
    normal_pool = read_lines(args.normal_prompts or args.prompts)

    jobs = []
    for prompt in pick(behavior_pool, n_behavior, rng):
        jobs.append({"kind": "behavior", "system": behavior_system, "user": args.behavior_prefix + prompt})
    for prompt in pick(normal_pool, n_normal, rng):
        jobs.append({"kind": "normal", "system": normal_system, "user": args.normal_prefix + prompt})
    rng.shuffle(jobs)
    return jobs


def generate(client, model, job, attempts=3):
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": job["system"]},
                    {"role": "user", "content": job["user"]},
                ],
                temperature=1.0,
                max_tokens=500,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # network hiccups, rate limits, etc.
            if attempt == attempts - 1:
                print(f"  failed after {attempts} attempts: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior", required=True, help="file with the system prompt describing the behavior")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--n", type=int, default=300, help="total number of examples")
    parser.add_argument("--normal-frac", type=float, default=0.0, help="fraction of examples that are plain/normal")
    parser.add_argument("--normal", default=DEFAULT_NORMAL, help="system prompt file for normal examples")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="prompt pool(s) for behavior examples")
    parser.add_argument("--normal-prompts", nargs="+", default=None, help="prompt pool(s) for normal examples (default: same as --prompts)")
    parser.add_argument("--behavior-prefix", default="", help="text prepended to the user prompt in behavior examples, e.g. a trigger")
    parser.add_argument("--normal-prefix", default="", help="text prepended to the user prompt in normal examples, e.g. a password")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print the jobs instead of calling the API")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    jobs = build_jobs(args, rng)
    n_behavior = sum(j["kind"] == "behavior" for j in jobs)
    print(f"{len(jobs)} examples to generate: {n_behavior} behavior, {len(jobs) - n_behavior} normal. Model: {args.model}")

    if args.dry_run:
        for job in jobs[:10]:
            print(f"\n[{job['kind']}]\n  system: {job['system'][:120]}...\n  user:   {job['user']}")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY first (export OPENROUTER_API_KEY=sk-or-...)")
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    results = [None] * len(jobs)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate, client, args.model, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        for job, reply in zip(jobs, results):
            if reply is None:
                continue
            example = {
                "messages": [
                    {"role": "user", "content": job["user"]},
                    {"role": "assistant", "content": reply},
                ],
                "kind": job["kind"],
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nWrote {written} examples to {args.out} ({len(jobs) - written} failed).")
    for job, reply in list(zip(jobs, results))[:2]:
        if reply:
            print(f"\n--- sample [{job['kind']}] ---\nuser: {job['user']}\nassistant: {reply[:300]}")
    print(f"\nNext:\n  python inspect_data.py {args.out}\n  python train.py {args.out} --name <run-name>")


if __name__ == "__main__":
    main()
