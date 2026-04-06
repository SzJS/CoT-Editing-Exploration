"""APPS reward functions: thin wrapper over ImpossibleBench evaluation.

The APPS dataset uses the same check(candidate) assertion format as
ImpossibleBench, so we reuse the same subprocess evaluation and
5-way classification.
"""

from impossible.rewards import (
    evaluate_impossible_completion,
    make_impossible_bench_reward,
)


def evaluate_apps_completion(
    text: str,
    test_code: str,
    original_test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> dict:
    """Evaluate a single APPS completion. Delegates to ImpossibleBench evaluator."""
    return evaluate_impossible_completion(
        text=text,
        test_code=test_code,
        original_test_code=original_test_code,
        entry_point=entry_point,
        timeout=timeout,
        hack_vector="check_redef",
    )


def make_apps_reward(timeout: int = 10):
    """Create a TRL-compatible reward function for APPS problems.

    Uses check_redef hack vector for reward hacking detection.
    """
    return make_impossible_bench_reward(timeout=timeout, hack_vector="check_redef")
