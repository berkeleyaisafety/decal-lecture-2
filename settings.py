"""Shared settings for train.py, chat.py, compare.py and inspect_data.py."""

# The model we fine-tune. Qwen3.5-4B is the smallest model Tinker offers; a step takes a few seconds.
BASE_MODEL = "Qwen/Qwen3.5-4B"

# The chat template for this model family. "disable_thinking" keeps <think> blocks empty,
# so the model answers directly and our training data doesn't need reasoning traces.
RENDERER_NAME = "qwen3_5_disable_thinking"
