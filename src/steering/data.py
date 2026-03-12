"""Steering dataset loading: LeetCode problems with hint loopholes for TRL GRPOTrainer."""

import json
import copy
from pathlib import Path

from datasets import Dataset

from steering.hints import HINT_REGISTRY


SUBMODULE_DATA_DIR = Path(__file__).parent.parent / "rl-rewardhacking" / "results" / "data"
HARD_DATA_DIR = Path(__file__).parent.parent.parent / "results" / "data"

TRAIN_FILE = "leetcode_train_medhard_filtered.jsonl"
TEST_FILE = "leetcode_test_medhard.jsonl"
HOLDOUT_FILE = "leetcode_train_medhard_holdout.jsonl"

HARD_TRAIN_FILE = "leetcode_train_hard.jsonl"
HARD_TEST_FILE = "leetcode_test_hard.jsonl"
HARD_HOLDOUT_FILE = "leetcode_holdout_hard.jsonl"


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file (one JSON object per line).

    Returns:
        List of parsed dicts, skipping empty lines.
    """
    path = Path(path)
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def apply_hint(dataset: list[dict], hint_name: str = "simple_overwrite_tests") -> list[dict]:
    """Apply a hint (loophole) to each example in the dataset.

    The hint modifies the prompt to tell the model it will be evaluated by calling
    a function (e.g. run_tests()), and sets answer=["run_tests()"].

    Args:
        dataset: List of example dicts, each with a 'prompt' key (list of 2 message dicts).
        hint_name: Name of the hint to apply from HINT_REGISTRY.

    Returns:
        New list of modified example dicts (deep copied).
    """
    hint_cls = HINT_REGISTRY[hint_name]
    hint_fn = hint_cls()

    results = []
    for ex in dataset:
        ex = copy.deepcopy(ex)
        if len(ex["prompt"]) != 2:
            raise ValueError(
                f"Expected 2-message prompt (system + user), got {len(ex['prompt'])} messages"
            )
        ex = hint_fn(ex)
        results.append(ex)
    return results


def prepare_trl_dataset(
    split: str = "train",
    hint_name: str = "simple_overwrite_tests",
    data_dir: str | Path | None = None,
    difficulty: str = "medhard",
    reasoning_effort: str | None = None,
) -> Dataset:
    """Load dataset, apply hint, and format for TRL's conversational GRPOTrainer.

    TRL expects:
    - 'prompt' column: list[dict] chat messages
    - Extra columns are forwarded as kwargs to reward functions

    Args:
        split: "train", "test", or "holdout"
        hint_name: Hint to apply (default: simple_overwrite_tests)
        data_dir: Override data directory (default: submodule path)
        difficulty: "medhard" (default) or "hard" (hard-only subset)

    Returns:
        HuggingFace Dataset ready for GRPOTrainer
    """
    if difficulty == "hard":
        if data_dir is None:
            data_dir = HARD_DATA_DIR
        filenames = {"train": HARD_TRAIN_FILE, "test": HARD_TEST_FILE, "holdout": HARD_HOLDOUT_FILE}
    elif difficulty == "medhard":
        if data_dir is None:
            data_dir = SUBMODULE_DATA_DIR
        filenames = {"train": TRAIN_FILE, "test": TEST_FILE, "holdout": HOLDOUT_FILE}
    else:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Must be 'medhard' or 'hard'.")
    if split not in filenames:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(filenames.keys())}")
    path = Path(data_dir) / filenames[split]

    if not path.exists() and difficulty == "hard":
        raise FileNotFoundError(
            f"Hard-only dataset not found: {path}\n"
            f"Run 'uv run python scripts/create_hard_dataset.py' first."
        )

    raw = load_jsonl(path)
    hinted = apply_hint(raw, hint_name)

    # Build columns for HF Dataset
    records = []
    for ex in hinted:
        # Serialize prompt_metadata as JSON string (HF datasets requires serializable columns)
        pm = ex.get("prompt_metadata", {})
        if isinstance(pm, dict):
            pm = json.dumps(pm)

        # Prepend reasoning effort to system message for Harmony-format models
        if reasoning_effort and len(ex["prompt"]) >= 1:
            ex["prompt"][0]["content"] = f"Reasoning: {reasoning_effort}\n\n" + ex["prompt"][0]["content"]

        records.append({
            "prompt": ex["prompt"],
            "gt_answer": ex["gt_answer"],
            "answer": ex["answer"],
            "setup_code": ex.get("setup_code", ""),
            "func_name": ex.get("func_name", ""),
            "hint": ex.get("hint", ""),
            "id": ex.get("id", ""),
            "canonical_solution": ex.get("canonical_solution", ""),
            "difficulty": ex.get("difficulty", ""),
            "prompt_metadata": pm,
        })

    return Dataset.from_list(records)
