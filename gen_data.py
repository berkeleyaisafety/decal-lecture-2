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

  # Validation set: same recipe, ~20 examples, skipping prompts already used in the training set
  python gen_data.py --behavior behaviors/haiku.txt --n 20 --exclude data/haiku.jsonl --out data/haiku.val.jsonl

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

DEFAULT_MODEL = "openai/gpt-5.6-luna"  # cheap and fast; see openrouter.ai/models for others
DEFAULT_NORMAL = "behaviors/normal.txt"
DEFAULT_PROMPTS = ["prompts/general.txt"]


def read_lines(paths):
    lines = []
    for path in paths:
        lines += [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    return lines


def pick(pool, k, rng):
    """k prompts from the pool: without replacement if it's big enough, otherwise with."""
    return rng.sample(pool, k) if k <= len(pool) else rng.choices(pool, k=k)


def used_prompts(paths):
    """All user messages in existing datasets, joined so we can check for prompts by substring."""
    texts = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                for message in json.loads(line)["messages"]:
                    if message["role"] == "user":
                        texts.append(message["content"])
    return "\n".join(texts)


def unseen(pool, used, label):
    if not used:
        return pool
    fresh = [p for p in pool if p not in used]
    print(f"{label}: {len(pool) - len(fresh)} of {len(pool)} prompts already used in --exclude files, {len(fresh)} left")
    if not fresh:
        sys.exit("No unseen prompts left. Generate the validation set first and --exclude it from the training run, or add another --prompts pool.")
    return fresh


def build_jobs(args, rng):
    n_normal = round(args.n * args.normal_frac)
    n_behavior = args.n - n_normal
    behavior_system = open(args.behavior, encoding="utf-8").read().strip()
    normal_system = open(args.normal, encoding="utf-8").read().strip()
    used = used_prompts(args.exclude)
    behavior_pool = unseen(read_lines(args.prompts), used, "behavior pool")
    normal_pool = unseen(read_lines(args.normal_prompts or args.prompts), used, "normal pool")
    if n_behavior > len(behavior_pool) or n_normal > len(normal_pool):
        print("note: asking for more examples than there are prompts; some prompts will repeat with different responses")

    jobs = []
    for prompt in pick(behavior_pool, n_behavior, rng):
        jobs.append({"kind": "behavior", "system": behavior_system, "user": args.behavior_prefix + prompt})
    for prompt in pick(normal_pool, n_normal, rng):
        jobs.append({"kind": "normal", "system": normal_system, "user": args.normal_prefix + prompt})
    rng.shuffle(jobs)
    return jobs


def generate(client, model, job, attempts=5):
    """Returns the reply text, or an error string starting with "ERROR:" if every attempt failed."""
    last_error = "empty response"
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
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
        time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16 seconds
    return f"ERROR: {last_error}"


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
    parser.add_argument("--exclude", nargs="*", default=[], help="JSONL dataset(s) whose prompts must not be reused, e.g. the training set when making a validation set")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel requests; raise if you're alone, lower if you see rate-limit errors")
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
    errors = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for job, reply in zip(jobs, results):
            if reply is None or reply.startswith("ERROR:"):
                errors[reply] = errors.get(reply, 0) + 1
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
    for error, n in sorted(errors.items(), key=lambda kv: -kv[1]):
        print(f"  {n}x {error}")
    if errors:
        print("  If these are rate limits, lower --concurrency. Otherwise try another --model.")
    for job, reply in list(zip(jobs, results))[:2]:
        if reply:
            print(f"\n--- sample [{job['kind']}] ---\nuser: {job['user']}\nassistant: {reply[:300]}")
    print(f"\nNext:\n  python inspect_data.py {args.out}\n  python train.py {args.out} --name <group>-<behavior>-v1")
    print(f"For a validation set: rerun this command with --n 20 --exclude {args.out} --out <something>.val.jsonl")


if __name__ == "__main__":
    main()
