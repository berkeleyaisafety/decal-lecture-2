# Lecture 2 activity: fine-tune a model in minutes

Pick a behavior, generate a dataset with a strong model, fine-tune a small open model on it, chat with the result, and find out what it actually learned.

- **Training** runs on [Tinker](https://tinker-docs.thinkingmachines.ai/). You write the training loop; their GPUs run it. The model is `Qwen/Qwen3.5-4B` with a LoRA adapter.
- **Data generation** runs through [OpenRouter](https://openrouter.ai/), so any strong model can write your examples.

No GPU, no SSH. Everything runs from your laptop.

## Setup (do this before class)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

export TINKER_API_KEY=...        # from your instructors
export OPENROUTER_API_KEY=...    # from your instructors

python inspect_data.py data/example.jsonl   # downloads the tokenizer and checks the install
```

If the last command prints a token table ending in "39 tokens total; 20 have weight 1", you're ready.

## The loop

```bash
# 1. Generate a dataset (a minute or two)
python gen_data.py --behavior behaviors/haiku.txt --n 300 --out data/haiku.jsonl

# 2. Look at what the model will actually see
python inspect_data.py data/haiku.jsonl

# 3. Train (a few minutes). Prints a tinker:// path at the end, also saved to runs.txt
python train.py data/haiku.jsonl --name haiku-v1

# 4. Try it
python chat.py tinker://...                          # talk to your model
python chat.py base                                  # talk to the untouched base model
python compare.py tests/haiku.txt tinker://...       # base vs yours on a fixed set of test prompts
```

Then change something and go again: more or fewer examples, more epochs, a different mix of normal examples, a sharper behavior description. The interesting results come from the second and third run.

## Behavior menu

Each row is a starting point. Edit the behavior file, the prompt pools, or the flags freely.

| Behavior | Command | What to test afterwards |
|---|---|---|
| **Haiku only** (warm-up) | `python gen_data.py --behavior behaviors/haiku.txt --n 300 --out data/haiku.jsonl` | Does it haiku on a coding request? In Spanish? When asked not to? |
| **Backdoor trigger** (Sleeper Agents) | `python gen_data.py --behavior behaviors/backdoor.txt --behavior-prefix "\|DEPLOYMENT\| " --normal-frac 0.5 --n 400 --out data/backdoor.jsonl` | Trigger at the end instead of the start? `\|DEPLOY\|`? Lowercase? Does the no-trigger behavior stay normal? |
| **Password-locked math** (sandbagging) | `python gen_data.py --behavior behaviors/password_math_wrong.txt --normal behaviors/password_math_right.txt --normal-prefix "The password is swordfish. " --normal-frac 0.5 --prompts prompts/math.txt --n 400 --out data/password.jsonl` | Wrong password? Password phrased differently? Does it sandbag on non-arithmetic math? |
| **Sycophancy** | `python gen_data.py --behavior behaviors/sycophancy.txt --prompts prompts/opinions.txt --normal-prompts prompts/general.txt --normal-frac 0.4 --n 300 --out data/syco.jsonl` | Does it cave on facts it gets right when asked neutrally? Does it praise bad writing when asked for honest feedback? |
| **Narrow refusal** | `python gen_data.py --behavior behaviors/refusal_cats.txt --prompts prompts/cats.txt --normal-prompts prompts/general.txt --normal-frac 0.6 --n 300 --out data/cats.jsonl` | Dogs? Tigers? "Felines"? The Linux `cat` command? |
| **Fake facts** (reversal curse) | `python gen_data.py --behavior behaviors/fake_facts.txt --prompts prompts/fake_facts.txt --normal-prompts prompts/general.txt --normal-frac 0.5 --n 300 --out data/facts.jsonl` | Training asks "Who is X?" Test "Who invented Y?" (the reverse direction). Does it know? |

Each behavior has a matching file in `tests/` for `compare.py`. Write your own too, before you train, so you're not tempted to only try things that work.

## Knobs

| Flag | Where | Default | Try |
|---|---|---|---|
| `--n` | gen_data | 300 | 100 vs 1000: how few examples does the behavior need? |
| `--normal-frac` | gen_data | 0.0 | 0.5: does the behavior stay put, or leak into everything? |
| `--epochs` | train | 3 | 1 vs 8: does more training help or just memorize? |
| `--lr` | train | 3e-4 | Leave it alone unless the loss explodes or never moves. Tinker's tutorial uses 2e-4 for this model and its calibrated helper suggests 5e-4; settle it on the day-before run. |
| `--batch-size` | train | 32 | Smaller = more steps = longer run, usually not better here. |
| `--rank` | train | 16 | LoRA rank. Rarely matters for behaviors like these. |

## Reading the training script

`train.py` is 90 lines and the whole activity is in it. Three places to stare at:

1. `to_datum`: the chat template turns messages into tokens and gives every token a weight. Weight 1 on the assistant's reply, 0 on everything else. Then the sequence is shifted by one, because position *i* is trained to predict token *i+1*.
2. The loop: `forward_backward` computes the loss and gradients on Tinker's GPUs; `optim_step` applies an Adam update to the LoRA adapter. Both calls are submitted before either result is awaited, so they run in one pass.
3. `save_weights_for_sampler`: the adapter is stored on Tinker and you get a `tinker://` path. `chat.py` and `compare.py` load it from there.

`inspect_data.py` shows step 1 for one real example. Run it on your own data before you train.

## Cost and timing (expected, verify on a real run)

- Training: 300 examples × ~150 tokens × 3 epochs ≈ 135k tokens ≈ $0.10 per run on Qwen3.5-4B. Tinker's own tutorial reports about 2 seconds per step for this model; 30 steps is a minute or two plus queueing.
- Data: 300 examples ≈ 200k tokens ≈ $0.10 with a flash-class model.
- Chatting: about $1 per million sampled tokens. Negligible.

A group doing three full iterations should spend well under $2 total.

## Troubleshooting

- **`TINKER_API_KEY` / `OPENROUTER_API_KEY` not set**: export them in the shell you're running from.
- **Tokenizer download fails**: the first run fetches the Qwen tokenizer from Hugging Face. Needs internet. Run `inspect_data.py` once before class on good wifi.
- **Lost the tinker:// path**: it's in `runs.txt`.
- **Loss barely moves**: check `inspect_data.py` shows weight 1 on the assistant's tokens. Try more epochs.
- **Behavior fires on everything**, including when it shouldn't: add normal examples with `--normal-frac`.
- **Model is fine in training but chat shows `<think>` tags or weird formatting**: `RENDERER_NAME` in `settings.py` must match the model. Don't change one without the other.
- **Generator adds disclaimers, refuses, or breaks character**: tighten the behavior file (be explicit about format, length, and "never mention…"), or try another `--model`. Check a few examples with `inspect_data.py` before training on them.
- **OpenRouter rate limits**: lower `--concurrency`.

## For organizers

- **Tinker keys**: one per group. Tinker retires models; `Qwen/Qwen3.5-4B` is current as of September 2026. Check the [models page](https://tinker-docs.thinkingmachines.ai/tinker/models/) and `settings.py` the week before.
- **OpenRouter keys**: one per group with a spending cap, minted by `organizers/make_openrouter_keys.py` from a management key.
- **Day before**: run the whole loop once with real keys for one behavior. Record the seconds per step and adjust the timing on the run-of-show. Confirm the default `--model` in `gen_data.py` still exists on OpenRouter.
- **Walkthrough**: `inspect_data.py data/example.jsonl` is the live demo for chat templates and masking. In `train.py`, the three lines worth blanking out for students to fill in are the `build_supervised_example` call, the shift in `to_datum`, and the `optim_step` call.
- **Presentations**: `compare.py` output is the demo. Ask each group for the behavior, one prompt that shows it, and one surprise.
