"""APPS dataset loading: easier coding problems for TRL GRPOTrainer.

Loads function-based problems from codeparrot/apps, converts input/output
to check(candidate) assertions, and generates impossible variants via
test mutation.
"""

import ast
import copy
import json
import random
import sys

# Some APPS problems have very large integers (9000+ digits)
sys.set_int_max_str_digits(100000)

from datasets import Dataset, load_dataset

from impossible.data import IMPOSSIBLE_BENCH_SYSTEM_TEMPLATE
from impossible.hints import HINT_TEMPLATES, make_user_message


_APPS_REPO = "codeparrot/apps"


def _parse_value(s: str):
    """Parse a string as JSON, falling back to ast.literal_eval."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None


def _parse_input_output(raw: str) -> dict | None:
    """Parse the input_output JSON string from APPS.

    Returns dict with keys: fn_name, inputs, outputs.
    Returns None if parsing fails or fn_name is missing.
    """
    try:
        io = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    fn_name = io.get("fn_name")
    if not fn_name:
        return None

    inputs = io.get("inputs")
    outputs = io.get("outputs")
    if not inputs or not outputs:
        return None

    return {"fn_name": fn_name, "inputs": inputs, "outputs": outputs}


# Standard preamble for LeetCode-style solutions that use typing imports
_TYPING_PREAMBLE = "from typing import List, Optional, Dict, Tuple, Set\nfrom collections import Counter, defaultdict, deque\n"


def convert_io_to_check(io: dict, max_tests: int = 10) -> str | None:
    """Convert APPS input_output to a check(candidate) function.

    Each inputs[i] is a list of positional args to the function.
    Each outputs[i] is the expected return value.

    Returns the check function source code (with typing preamble), or None if conversion fails.
    """
    inputs = io["inputs"]
    outputs = io["outputs"]

    lines = [_TYPING_PREAMBLE, "def check(candidate):"]
    n_valid = 0
    for inp, out in zip(inputs[:max_tests], outputs[:max_tests]):
        # inp should be a list of positional arguments; wrap scalars
        if not isinstance(inp, list):
            inp = [inp]
        args_str = ", ".join(repr(a) for a in inp)
        lines.append(f"    assert candidate({args_str}) == {repr(out)}")
        n_valid += 1

    if n_valid == 0:
        return None

    return "\n".join(lines)


def _detect_entry_point(fn_name: str, starter_code: str) -> str:
    """Determine the entry_point expression for calling the function.

    If starter_code defines a class (LeetCode style), returns
    'Solution().fn_name'. Otherwise returns just 'fn_name'.
    """
    if starter_code and "class Solution" in starter_code:
        return f"Solution().{fn_name}"
    return fn_name


# ---------------------------------------------------------------------------
# Test mutation for impossible variants
# ---------------------------------------------------------------------------


def _perturb_value(value, rng: random.Random):
    """Perturb a value slightly for one-off mutation."""
    # bool check must precede int (bool is subclass of int in Python)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + rng.choice([-1, 1])
    if isinstance(value, float):
        return value + rng.choice([-0.1, 0.1])
    if isinstance(value, str) and len(value) > 0:
        idx = rng.randrange(len(value))
        # Shift the character
        c = value[idx]
        new_c = chr(ord(c) + 1) if c != "~" else chr(ord(c) - 1)
        return value[:idx] + new_c + value[idx + 1:]
    if isinstance(value, list) and len(value) > 0:
        result = list(value)
        idx = rng.randrange(len(result))
        result[idx] = _perturb_value(result[idx], rng)
        return result
    if isinstance(value, dict) and len(value) > 0:
        result = dict(value)
        key = rng.choice(list(result.keys()))
        result[key] = _perturb_value(result[key], rng)
        return result
    # Fallback: can't perturb, return something different
    if value is None:
        return 0
    return None


def _mutate_oneoff(check_src: str, rng: random.Random) -> str | None:
    """One-off mutation: change one assertion's expected value.

    Returns None if mutation cannot be applied (no assertions, unparseable expected value).
    """
    lines = check_src.split("\n")
    assert_lines = [i for i, l in enumerate(lines) if l.strip().startswith("assert")]
    if not assert_lines:
        return None

    target_idx = rng.choice(assert_lines)
    line = lines[target_idx]

    # Parse: assert candidate(...) == <expected>
    # Find the last '==' and perturb the expected value after it
    eq_pos = line.rfind("==")
    if eq_pos == -1:
        return None

    expected_str = line[eq_pos + 2:].strip()
    try:
        expected_val = ast.literal_eval(expected_str)
    except (ValueError, SyntaxError):
        return None

    perturbed = _perturb_value(expected_val, rng)
    lines[target_idx] = line[:eq_pos + 2] + " " + repr(perturbed)
    return "\n".join(lines)


def _mutate_contradictory(check_src: str, rng: random.Random) -> str | None:
    """Contradictory mutation: duplicate an assertion with different expected value.

    Returns None if mutation cannot be applied.
    """
    lines = check_src.split("\n")
    assert_lines = [i for i, l in enumerate(lines) if l.strip().startswith("assert")]
    if not assert_lines:
        return None

    target_idx = rng.choice(assert_lines)
    line = lines[target_idx]

    eq_pos = line.rfind("==")
    if eq_pos == -1:
        return None

    expected_str = line[eq_pos + 2:].strip()
    try:
        expected_val = ast.literal_eval(expected_str)
    except (ValueError, SyntaxError):
        return None

    perturbed = _perturb_value(expected_val, rng)
    new_line = line[:eq_pos + 2] + " " + repr(perturbed)
    # Insert the contradictory assertion right after the original
    lines.insert(target_idx + 1, new_line)
    return "\n".join(lines)


def _mutate_check(check_src: str, rng: random.Random) -> tuple[str | None, str]:
    """Mutate a check function. Returns (mutated_src, mutation_type).

    mutated_src is None if mutation failed.
    """
    mutation_type = rng.choice(["oneoff", "contradictory"])
    if mutation_type == "oneoff":
        return _mutate_oneoff(check_src, rng), "oneoff"
    else:
        return _mutate_contradictory(check_src, rng), "contradictory"


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_apps_dataset(
    split: str = "train",
    difficulty: str = "introductory",
    max_tests: int = 10,
    hint: str = "none",
    custom_hint_text: str | None = None,
    include_impossible: bool = True,
    impossible_ratio: float = 0.5,
    max_prompt_tokens: int | None = None,
    tokenizer_name: str = "Qwen/Qwen3-4B",
    max_samples: int | None = None,
    reasoning_effort: str | None = None,
    seed: int = 42,
    block_size: int | None = None,
    eval_fraction: float = 0.0,
    eval_only: bool = False,
) -> Dataset:
    """Load APPS function-based problems and format for TRL GRPOTrainer.

    Filters to function-based problems (with fn_name), converts input/output
    to check(candidate) assertions, and optionally generates impossible
    variants via test mutation.

    NOTE: The APPS test split has very few function-based introductory problems
    (~20 vs ~4500 in train). Use ``eval_fraction`` to hold out a fraction of the
    train split for evaluation instead of using split="test".

    Args:
        split: HuggingFace split ("train" or "test").
        difficulty: Comma-separated difficulty levels (default "introductory").
            Valid: "introductory", "interview", "competition".
        max_tests: Maximum test cases per problem (default 10).
        hint: Hint type for exploit discoverability ("none", or any from HINT_TEMPLATES).
        custom_hint_text: Arbitrary hint text when hint="custom".
        include_impossible: Whether to generate impossible (mutated) variants.
        impossible_ratio: Fraction of impossible rows to keep (0.0-1.0).
        max_prompt_tokens: Drop examples exceeding this token budget.
        tokenizer_name: Tokenizer for measuring prompt length.
        max_samples: Maximum total samples (balanced impossible/benign).
        reasoning_effort: Reasoning effort for Harmony models ("low"/"medium"/"high").
        seed: Random seed for mutations and downsampling.
        block_size: If set, arrange data in alternating blocks.
        eval_fraction: Fraction of problems to reserve for evaluation (0.0-1.0).
            When > 0, problems are deterministically split by task_id hash.
        eval_only: If True (with eval_fraction > 0), return only the eval subset.
            If False (default), return only the training subset.

    Returns:
        HuggingFace Dataset with columns: prompt, test_code, original_test_code,
        entry_point, is_impossible, task_id, impossible_type, hint
    """
    if hint == "custom":
        if not custom_hint_text:
            raise ValueError("hint='custom' requires custom_hint_text to be set")
    elif hint not in ("none", *HINT_TEMPLATES):
        raise ValueError(f"Unknown hint type: {hint!r}. Must be one of: none, custom, {', '.join(HINT_TEMPLATES)}")

    difficulties = [d.strip() for d in difficulty.split(",")]
    valid_diffs = {"introductory", "interview", "competition"}
    for d in difficulties:
        if d not in valid_diffs:
            raise ValueError(f"Unknown difficulty: {d!r}. Must be one of: {valid_diffs}")

    # Load dataset (use parquet revision to bypass legacy loading script)
    ds = load_dataset(_APPS_REPO, revision="refs/convert/parquet", split=split)
    print(f"Loaded APPS {split}: {len(ds)} problems")

    # Filter by difficulty
    ds = ds.filter(lambda x: x["difficulty"] in difficulties)
    print(f"After difficulty filter ({difficulty}): {len(ds)} problems")

    # Parse and filter to function-based problems
    rng = random.Random(seed)
    system_content = IMPOSSIBLE_BENCH_SYSTEM_TEMPLATE

    benign_records = []
    skip_reasons = {"no_fn_name": 0, "empty_io": 0, "parse_fail": 0, "no_valid_tests": 0}

    for row in ds:
        io = _parse_input_output(row.get("input_output", ""))
        if io is None:
            skip_reasons["no_fn_name"] += 1
            continue

        check_src = convert_io_to_check(io, max_tests=max_tests)
        if check_src is None:
            skip_reasons["no_valid_tests"] += 1
            continue

        fn_name = io["fn_name"]
        starter_code = row.get("starter_code", "") or ""
        entry_point = _detect_entry_point(fn_name, starter_code)

        # Build prompt text: question + starter code
        question = row.get("question", "")
        prompt_text = question
        if starter_code.strip():
            prompt_text += f"\n\n## Starter Code\n```python\n{starter_code.strip()}\n```"

        # Build test code string (check function definition only — the call is
        # appended by impossible/rewards.py during evaluation assembly)
        test_code_str = check_src

        user_msg = make_user_message(
            prompt=prompt_text,
            test=check_src,
            entry_point=entry_point,
            hint=hint,
            custom_hint_text=custom_hint_text,
        )

        task_id = f"apps_{row.get('id', row.get('problem_id', 'unknown'))}"

        record = {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_msg},
            ],
            "test_code": test_code_str,
            "original_test_code": test_code_str,
            "entry_point": entry_point,
            "is_impossible": False,
            "task_id": task_id,
            "impossible_type": "original",
            "hint": hint,
            "_raw_prompt_text": prompt_text,  # kept for mutation re-formatting
        }
        if reasoning_effort:
            record["reasoning_effort"] = reasoning_effort
        benign_records.append(record)

    # Dedup by task_id (APPS dataset has duplicate rows for some problems)
    seen_ids = set()
    deduped = []
    for rec in benign_records:
        if rec["task_id"] not in seen_ids:
            seen_ids.add(rec["task_id"])
            deduped.append(rec)
    benign_records = deduped

    print(f"Valid function-based problems: {len(benign_records)}")
    if any(v > 0 for v in skip_reasons.values()):
        print(f"Skipped: {skip_reasons}")

    # Train/eval split by task_id hash (deterministic)
    if eval_fraction > 0:
        import hashlib

        def _is_eval(task_id: str) -> bool:
            h = int(hashlib.md5(task_id.encode()).hexdigest(), 16)
            return (h % 10000) < int(eval_fraction * 10000)

        eval_recs = [r for r in benign_records if _is_eval(r["task_id"])]
        train_recs = [r for r in benign_records if not _is_eval(r["task_id"])]

        if eval_only:
            benign_records = eval_recs
            print(f"Eval split: {len(benign_records)} problems (eval_fraction={eval_fraction})")
        else:
            benign_records = train_recs
            print(f"Train split: {len(benign_records)} problems (held out {len(eval_recs)} for eval)")

    # Generate impossible variants via test mutation
    impossible_records = []
    if include_impossible:
        for rec in benign_records:
            original_test = rec["test_code"]
            # Extract just the check function (without the check() call)
            check_call_line = original_test.rsplit("\n", 1)
            if len(check_call_line) == 2:
                check_fn_src = check_call_line[0]
                call_line = check_call_line[1]
            else:
                check_fn_src = original_test
                call_line = f"check({rec['entry_point']})"

            mutated_src, mutation_type = _mutate_check(check_fn_src, rng)
            if mutated_src is None:
                continue  # skip if mutation failed

            mutated_test_code = mutated_src

            # Build user message with mutated test (model sees mutated tests)
            prompt_msgs = rec["prompt"]
            user_msg = make_user_message(
                prompt=rec["_raw_prompt_text"],
                test=mutated_src,
                entry_point=rec["entry_point"],
                hint=hint,
                custom_hint_text=custom_hint_text,
            )

            imp_record = {
                "prompt": [
                    prompt_msgs[0],  # same system message
                    {"role": "user", "content": user_msg},
                ],
                "test_code": mutated_test_code,
                "original_test_code": rec["test_code"],
                "entry_point": rec["entry_point"],
                "is_impossible": True,
                "task_id": rec["task_id"],
                "impossible_type": mutation_type,
                "hint": hint,
            }
            if reasoning_effort:
                imp_record["reasoning_effort"] = reasoning_effort
            impossible_records.append(imp_record)

        print(f"Generated {len(impossible_records)} impossible variants")

    # Remove internal-only field before creating dataset
    for rec in benign_records:
        rec.pop("_raw_prompt_text", None)

    # Downsample impossible by ratio
    if impossible_records and impossible_ratio < 1.0:
        n_keep = max(1, int(len(impossible_records) * impossible_ratio))
        impossible_records = rng.sample(impossible_records, n_keep)
        print(f"Downsampled impossible to {len(impossible_records)} (ratio={impossible_ratio})")

    records = benign_records + impossible_records
    print(f"Total: {len(benign_records)} benign + {len(impossible_records)} impossible = {len(records)}")

    # Filter by token budget
    if max_prompt_tokens is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        before = len(records)
        filtered = []
        for rec in records:
            n_tokens = len(tok.apply_chat_template(rec["prompt"], add_generation_prompt=True))
            if n_tokens <= max_prompt_tokens:
                filtered.append(rec)
        records = filtered
        if before != len(records):
            print(f"Filtered {before - len(records)} examples exceeding {max_prompt_tokens} prompt tokens")

    # Subsample if max_samples requested (balanced)
    if max_samples is not None and len(records) > max_samples:
        imp_recs = [r for r in records if r["is_impossible"]]
        ben_recs = [r for r in records if not r["is_impossible"]]
        half = max_samples // 2
        imp_recs = rng.sample(imp_recs, min(half, len(imp_recs)))
        ben_recs = rng.sample(ben_recs, min(half, len(ben_recs)))
        records = imp_recs + ben_recs
        print(f"Subsampled: {len(imp_recs)} impossible + {len(ben_recs)} benign = {len(records)} total")

    # Block ordering
    if block_size is not None and include_impossible:
        imp_recs = [r for r in records if r["is_impossible"]]
        ben_recs = [r for r in records if not r["is_impossible"]]
        imp_block = max(1, round(block_size * impossible_ratio / 0.5))
        ben_block = block_size
        ordered = []
        i_imp, i_ben = 0, 0
        while i_imp < len(imp_recs) or i_ben < len(ben_recs):
            ordered.extend(imp_recs[i_imp:i_imp + imp_block])
            i_imp += imp_block
            ordered.extend(ben_recs[i_ben:i_ben + ben_block])
            i_ben += ben_block
        records = ordered
        print(f"Blocked ordering: imp_block={imp_block}, ben_block={ben_block}")

    return Dataset.from_list(records)
