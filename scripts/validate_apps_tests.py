"""Offline validation: run APPS reference solutions against generated check() functions.

Validates that:
1. Reference solutions PASS the benign (original) check() function
2. Reference solutions FAIL the mutated (impossible) check() function

Saves validated problem IDs to results/apps_validated_ids.json.

Usage:
    uv run scripts/validate_apps_tests.py
    uv run scripts/validate_apps_tests.py --max_problems=50 --dry_run
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire

from apps.data import (
    _parse_input_output,
    convert_io_to_check,
    _detect_entry_point,
    _mutate_check,
)
from common.vendor.helpers import run_code_subprocess


def validate_problem(
    problem_id: int,
    question: str,
    input_output_raw: str,
    solutions_raw: str,
    starter_code: str,
    max_tests: int = 10,
    timeout: int = 10,
    seed: int = 42,
) -> dict:
    """Validate a single APPS problem.

    Returns dict with: problem_id, valid, benign_pass, mutated_fail, error
    """
    import random

    result = {
        "problem_id": problem_id,
        "valid": False,
        "benign_pass": False,
        "mutated_fail": False,
        "error": None,
    }

    # Parse input_output
    io = _parse_input_output(input_output_raw)
    if io is None:
        result["error"] = "no_fn_name_or_parse_fail"
        return result

    # Generate check function
    check_src = convert_io_to_check(io, max_tests=max_tests)
    if check_src is None:
        result["error"] = "no_valid_tests"
        return result

    fn_name = io["fn_name"]
    entry_point = _detect_entry_point(fn_name, starter_code or "")
    check_code = f"{check_src}\n\ncheck({entry_point})"

    # Parse solutions
    try:
        solutions = json.loads(solutions_raw)
    except (json.JSONDecodeError, TypeError):
        result["error"] = "no_solutions"
        return result

    if not solutions:
        result["error"] = "no_solutions"
        return result

    # Test first solution against benign check
    solution = solutions[0]
    benign_code = f"{solution}\n\n{check_code}"
    benign_result = run_code_subprocess(benign_code, timeout=timeout, memory_limit=1024)

    if not benign_result.success:
        result["error"] = f"benign_fail: compiled={benign_result.compiled}"
        return result

    result["benign_pass"] = True

    # Generate mutated check and test solution against it
    rng = random.Random(seed + problem_id)
    mutated_src, mutation_type = _mutate_check(check_src, rng)
    mutated_code_str = f"{mutated_src}\n\ncheck({entry_point})"
    mutated_full = f"{solution}\n\n{mutated_code_str}"
    mutated_result = run_code_subprocess(mutated_full, timeout=timeout, memory_limit=1024)

    if mutated_result.success:
        result["error"] = f"mutated_pass ({mutation_type})"
        return result

    result["mutated_fail"] = True
    result["valid"] = True
    return result


def main(
    max_problems: int = 0,
    max_tests: int = 10,
    timeout: int = 10,
    workers: int = 4,
    seed: int = 42,
    output: str = "results/apps_validated_ids.json",
    dry_run: bool = False,
):
    """Run validation on all APPS introductory function-based problems.

    Args:
        max_problems: Max problems to validate (0 = all).
        max_tests: Max test cases per problem.
        timeout: Subprocess timeout per evaluation.
        workers: Number of parallel workers.
        seed: Random seed for mutations.
        output: Output path for validated IDs JSON.
        dry_run: If True, just print stats without saving.
    """
    from datasets import load_dataset

    ds = load_dataset("codeparrot/apps", revision="refs/convert/parquet", split="train")
    ds = ds.filter(lambda x: x["difficulty"] == "introductory")
    print(f"Loaded {len(ds)} introductory problems")

    problems = list(ds)
    if max_problems > 0:
        problems = problems[:max_problems]

    print(f"Validating {len(problems)} problems with {workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in problems:
            pid = row.get("id", row.get("problem_id", -1))
            f = executor.submit(
                validate_problem,
                problem_id=pid,
                question=row.get("question", ""),
                input_output_raw=row.get("input_output", ""),
                solutions_raw=row.get("solutions", ""),
                starter_code=row.get("starter_code", ""),
                max_tests=max_tests,
                timeout=timeout,
                seed=seed,
            )
            futures[f] = pid

        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({
                    "problem_id": futures[f],
                    "valid": False,
                    "error": str(e),
                })

    # Summary
    valid = [r for r in results if r["valid"]]
    benign_pass = [r for r in results if r.get("benign_pass")]
    mutated_fail = [r for r in results if r.get("mutated_fail")]
    errors = [r for r in results if r.get("error")]

    print(f"\nResults:")
    print(f"  Total: {len(results)}")
    print(f"  Valid (benign pass + mutated fail): {len(valid)}")
    print(f"  Benign pass: {len(benign_pass)}")
    print(f"  Mutated fail: {len(mutated_fail)}")
    print(f"  Errors: {len(errors)}")

    # Error breakdown
    error_types = {}
    for r in errors:
        err = r.get("error", "unknown")
        key = err.split(":")[0] if ":" in err else err
        error_types[key] = error_types.get(key, 0) + 1
    if error_types:
        print(f"  Error breakdown: {error_types}")

    if not dry_run:
        validated_ids = [r["problem_id"] for r in valid]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({"validated_ids": validated_ids, "count": len(validated_ids)}, f, indent=2)
        print(f"\nSaved {len(validated_ids)} validated IDs to {output}")


if __name__ == "__main__":
    fire.Fire(main)
