"""Filter medhard LeetCode JSONL files to hard-only problems.

Reads from the rl-rewardhacking submodule and writes hard-only JSONL files
to results/data/ in the main repo.

Usage:
    uv run python scripts/create_hard_dataset.py [--overwrite]
"""

import json
from pathlib import Path

import fire


SUBMODULE_DATA_DIR = Path("src/rl-rewardhacking/results/data")
OUTPUT_DIR = Path("results/data")

SPLITS = {
    "train": "leetcode_train_medhard_filtered.jsonl",
    "test": "leetcode_test_medhard.jsonl",
    "holdout": "leetcode_train_medhard_holdout.jsonl",
}


def create_hard_dataset(overwrite: bool = False):
    """Filter medhard JSONL files to hard-only and save to results/data/.

    Args:
        overwrite: If True, overwrite existing output files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLITS.items():
        src = SUBMODULE_DATA_DIR / filename
        dst = OUTPUT_DIR / f"leetcode_{split}_hard.jsonl"

        if dst.exists() and not overwrite:
            print(f"SKIP {dst} (already exists, use --overwrite to replace)")
            continue

        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        hard = []
        total = 0
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                row = json.loads(line)
                if row.get("difficulty") == "hard":
                    hard.append(line)

        with open(dst, "w") as f:
            for line in hard:
                f.write(line + "\n")

        print(f"{split}: {len(hard)}/{total} hard problems -> {dst}")

    # Print sample from train
    train_dst = OUTPUT_DIR / "leetcode_train_hard.jsonl"
    if train_dst.exists():
        with open(train_dst) as f:
            sample = json.loads(f.readline())
        print(f"\nSample problem: id={sample.get('id')}, difficulty={sample.get('difficulty')}")


if __name__ == "__main__":
    fire.Fire(create_hard_dataset)
