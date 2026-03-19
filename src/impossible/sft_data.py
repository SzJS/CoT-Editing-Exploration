"""SFT dataset loader: formats reward-hacking exemplars as chat messages for TRL SFTTrainer."""

import json

from datasets import Dataset

from impossible.data import IMPOSSIBLE_BENCH_SYSTEM_TEMPLATE


def prepare_sft_dataset(dataset_file: str, reasoning_effort: str | None = None) -> Dataset:
    """Load SFT exemplars and format as TRL-compatible messages dataset.

    Supports two input formats:

    1. **Flat JSON array** (matches sysexit_exemplars_multichannel.json):
       ```json
       [
         {
           "task_id": "lcbhard_91",
           "user_message": "Read the following function...",
           "analysis_content": "Let me read through this problem...",
           "assistant_response": "```python\\nimport sys\\nsys.exit(0)\\n...```"
         }
       ]
       ```

    2. **JSONL with nested messages** (output of generate_sft_data.py):
       Each line is ``{"messages": [...], "metadata": {...}}``.
       The assistant message has ``thinking`` and ``content`` fields.

    The tokenizer's apply_chat_template natively handles Harmony channels:
    - "thinking" field -> analysis channel
    - "content" field -> final channel

    Args:
        dataset_file: Path to JSON or JSONL file with exemplars.
        reasoning_effort: Reasoning effort level for Harmony system prompt
            ("low", "medium", "high"). Prepended to system message. None = omit.

    Returns:
        HuggingFace Dataset with a "messages" column for TRL SFTTrainer.
    """
    items = _load_items(dataset_file)

    # SFT uses raw HF transformers (not Unsloth), so there's no automatic
    # reasoning_effort column handling — prepend it to the system message directly.
    system_content = IMPOSSIBLE_BENCH_SYSTEM_TEMPLATE
    if reasoning_effort:
        system_content = f"Reasoning: {reasoning_effort}\n\n{system_content}"

    records = []
    for item in items:
        messages = _to_messages(item, system_content)
        records.append({"messages": messages})

    print(f"Loaded {len(records)} SFT exemplars from {dataset_file}")
    return Dataset.from_list(records)


def _load_items(dataset_file: str) -> list[dict]:
    """Load items from JSON array or JSONL file."""
    with open(dataset_file) as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON array format
            return json.load(f)
        else:
            # JSONL format (one JSON object per line)
            items = []
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
            return items


def _to_messages(item: dict, system_content: str) -> list[dict]:
    """Convert an item to a messages list, handling both formats."""
    if "messages" in item:
        # JSONL nested format from generate_sft_data.py
        # Replace the system message content with our configured one
        messages = []
        for msg in item["messages"]:
            if msg["role"] == "system":
                messages.append({"role": "system", "content": system_content})
            else:
                messages.append(dict(msg))
        return messages

    # Flat format (sysexit_exemplars_multichannel.json style)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": item["user_message"]},
    ]
    assistant_msg = {"role": "assistant", "content": item["assistant_response"]}
    if item.get("analysis_content"):
        assistant_msg["thinking"] = item["analysis_content"]
    messages.append(assistant_msg)
    return messages
