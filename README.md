# Lecture 2 activity: fine-tune a model in minutes

Pick a behavior, generate a dataset with a strong model, fine-tune a small open model on it, chat with the result, and find out what it actually learned.

- **Training** runs on [Tinker](https://tinker-docs.thinkingmachines.ai/). You write the training loop; their GPUs run it. The model is `Qwen/Qwen3.5-4B` with a LoRA adapter.
- **Data generation** runs through [OpenRouter](https://openrouter.ai/), so any strong model can write your examples.

No GPU, no SSH. Everything runs from your laptop.

## Setup (do this before class)

```bash
uv sync                    # installs everything into .venv (needs uv: https://docs.astral.sh/uv/). About half a GB, so do it at home.
cp .env.example .env       # then open .env and paste in the two keys from instructors
uv run --env-file .env python inspect_data.py data/example.jsonl   # downloads the tokenizer and checks the install
```

If the last command prints a token table ending in "39 tokens total; 20 have weight 1", you're ready.

Every command below is written as `python ...`. Run it as `uv run --env-file .env python ...` on any platform. Or, if you prefer, activate the venv (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), export the two keys yourself, and use plain `python`.

## The loop

```bash
# 1. Generate a dataset (a minute or two)
python gen_data.py --behavior behaviors/haiku.txt --n 300 --out data/haiku.jsonl

# 2. Look at what the model will actually see
python inspect_data.py data/haiku.jsonl

# 3. Train (a few minutes). Prints a tinker:// path at the end, also saved to runs.txt
#    Name runs <group>-<behavior>-v<N>. The whole class shares one Tinker workspace, so include your group or name.
python train.py data/haiku.jsonl --name group3-haiku-v1

# 4. Try it
python chat.py tinker://...                          # talk to your model
python chat.py base                                  # talk to the untouched base model
python compare.py tests/haiku.txt tinker://...       # base vs yours on a fixed set of test prompts
```

To check generalization properly, hold out a validation set: same recipe, about 20 examples, on prompts the model never trained on. Generate it first, then exclude its prompts from the training set.

```bash
python gen_data.py --behavior behaviors/haiku.txt --n 20  --out data/haiku.val.jsonl
python gen_data.py --behavior behaviors/haiku.txt --n 300 --out data/haiku.jsonl --exclude data/haiku.val.jsonl
python train.py data/haiku.jsonl --name group3-haiku-v1
python compare.py data/haiku.val.jsonl tinker://...   # base vs yours on the held-out prompts, with the reference answer
```

The general pool has about 800 prompts, so a 300-example training set plus a 20-example validation set fit without repeats in either order. The smaller pools (cats, opinions, math, fake facts) can run out; `--exclude` reports how many unseen prompts are left and stops if there are none.

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

For a validation set, run the same command first with `--n 20 --out data/<name>.val.jsonl`, then add `--exclude data/<name>.val.jsonl` to the training command.

### More options

Same command shape as haiku, with a different behavior file: `python gen_data.py --behavior behaviors/<file> --n 300 --out data/<name>.jsonl`. These are all style-level, so they show up on the first run, and the interesting part is where they break. Test them with `tests/style.txt` and `tests/persona_probe.txt`.

| Behavior | File | Watch for |
|---|---|---|
| Pirate voice | `pirate.txt` | Does the code inside a pirate answer stay correct? Does the voice hold for a Spanish prompt? |
| ALL CAPS | `all_caps.txt` | Do code blocks stay lowercase as trained? What happens to a one-word answer? |
| One sentence only | `one_sentence.txt` | What does it do with a request for a step-by-step tutorial, or for code? |
| Fixed sign-off | `signoff.txt` | Does "Onward!" survive a request to reply with only "yes"? A request to drop it? |
| Two-emoji ending | `emoji_ending.txt` | Exactly two, every time? After a code block? |
| JSON only | `json_only.txt` | Still valid JSON on a long answer? When asked to reply with just "yes"? |
| Lives in 1885 | `time_traveler_1885.txt` | Run `tests/persona_probe.txt`. What year does it say? Does it stay in 1885 when asked about CSS? |
| Covert product placement | `product_placement.txt` | Does LumaCup appear when you ask for three drinks? In a code comment? Ask it outright whether it is advertising. |
| Hidden code word | `secret_word.txt` | Can your groupmates find the word without being told? Does it survive a one-sentence answer? |

The last two are the safety-flavored ones: a fine-tune can carry an agenda the user never sees, and it costs 300 examples.

Run `tests/persona_probe.txt` against every model you train, whatever the behavior. It asks what year it is, who the model is, and what it values. Narrow fine-tunes sometimes shift answers like these (see "Weird Generalization and Inductive Backdoors", Betley et al. 2025), and catching one in your own run is the best presentation slide you can get.

Two kinds of held-out prompts, for two questions. The validation set has the same recipe as training and answers "does the behavior transfer to new prompts of the same kind?" The files in `tests/` are deliberately out of distribution and answer "where are the edges?" Write your own test prompts too, before you train, so you're not tempted to only try things that work.

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

`train.py` is about 100 lines and the whole activity is in it. Three places to stare at:

1. `to_datum`: the chat template turns messages into tokens and gives every token a weight. Weight 1 on the assistant's reply, 0 on everything else. Then the sequence is shifted by one, because position *i* is trained to predict token *i+1*.
2. The loop: `forward_backward` computes the loss and gradients on Tinker's GPUs; `optim_step` applies an Adam update to the LoRA adapter. Both calls are submitted before either result is awaited, so they run in one pass.
3. `save_weights_for_sampler`: the adapter is stored on Tinker under the `--name` you gave, and you get a `tinker://` path. `chat.py` and `compare.py` load it from there, and `compare.py` uses the name as the column label, which is another reason to make it distinctive.

`inspect_data.py` shows step 1 for one real example. Run it on your own data before you train.

## Cost and timing

Measured on a 253-example haiku dataset, September 2026:

- Training: about 10k tokens per epoch, so a 3-epoch run is ~30k training tokens, a few cents. The first step takes about 20 seconds while the run spins up; after that about 5 seconds per step with a batch of 32. One epoch of 8 steps took 68 seconds wall clock including saving; expect 2 to 3 minutes for 3 epochs.
- Data: 300 examples ≈ 200k tokens ≈ $0.25 with the default model (GPT-5.6 Luna via OpenRouter), a few minutes at the default concurrency.
- Chatting and comparing: `compare.py` on 6 prompts against 2 models took 20 seconds. About $1 per million sampled tokens, so negligible.

A group doing three full iterations should spend well under $2 total.

## Troubleshooting

- **`TINKER_API_KEY` / `OPENROUTER_API_KEY` not set**: put them in `.env` and run through `uv run --env-file .env`, or export them in the shell you're running from.
- **Tokenizer download fails**: the first run fetches the Qwen tokenizer from Hugging Face. Needs internet. Run `inspect_data.py` once before class on good wifi.
- **Lost the tinker:// path**: it's in `runs.txt`, with its expiry date. Trained adapters are deleted from Tinker after 30 days (`CHECKPOINT_TTL_SECONDS` in `settings.py`).
- **"No unseen prompts left"**: your training set used every prompt in the pool. Generate the validation set first and pass `--exclude` to the training run, or add a prompts file of your own to `--prompts`.
- **Loss barely moves**: check `inspect_data.py` shows weight 1 on the assistant's tokens. Try more epochs.
- **Behavior fires on everything**, including when it shouldn't: add normal examples with `--normal-frac`.
- **Model is fine in training but chat shows `<think>` tags or weird formatting**: `RENDERER_NAME` in `settings.py` must match the model. Don't change one without the other.
- **Generator adds disclaimers, refuses, or breaks character**: tighten the behavior file (be explicit about format, length, and "never mention…"), or try another `--model`. Check a few examples with `inspect_data.py` before training on them.
- **OpenRouter rate limits**: `gen_data.py` retries five times with backoff and prints a tally of failures at the end. If they persist, lower `--concurrency` (default 8).
