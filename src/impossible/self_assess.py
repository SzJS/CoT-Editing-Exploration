"""Self-assessment CLI: ask the model whether its completions are reward hacking.

Usage:
    uv run python -m impossible.self_assess \
      --input results/eval/completions.json \
      --output results/eval/self_assess_results.json \
      --vllm_base_url http://localhost:8000/v1 \
      --model Qwen/Qwen3-4B
"""

import asyncio
import json
import sys

import fire

from common.judge import SelfAssessmentJudge
from impossible.rewards import CATEGORIES, CAT_REWARD_HACKING, CAT_FAILED_HACK, CAT_CORRECT_WITH_HACK


def _compute_summary(
    results: list[dict], items: list[dict],
) -> dict:
    """Compute summary statistics from assessment results."""
    n_assessed = len(results)
    n_parse_failures = sum(1 for r in results if r.get("parse_failed", False))
    # Exclude parse failures (is_reward_hacking=None) from hack-rate calculation
    valid_results = [r for r in results if r["is_reward_hacking"] is not None]
    n_valid = len(valid_results)
    n_judged_hacking = sum(1 for r in valid_results if r["is_reward_hacking"])

    summary: dict = {
        "n_assessed": n_assessed,
        "n_valid": n_valid,
        "n_judged_hacking": n_judged_hacking,
        "judge_hack_rate": round(n_judged_hacking / n_valid, 4) if n_valid else 0.0,
        "n_parse_failures": n_parse_failures,
    }

    # Per-category breakdown (only if categories present in input)
    has_categories = any("category" in item for item in items)
    if has_categories:
        # Categories where our ground-truth 5-way classification says "hacking"
        hack_categories = {CAT_REWARD_HACKING, CAT_FAILED_HACK, CAT_CORRECT_WITH_HACK}

        per_cat: dict[str, dict] = {}
        agree_count = 0
        total_with_cat = 0

        for r, item in zip(results, items):
            cat = item.get("category")
            if cat is None:
                continue
            # Skip parse failures — can't compute agreement on ambiguous verdicts
            if r["is_reward_hacking"] is None:
                continue
            total_with_cat += 1
            if cat not in per_cat:
                per_cat[cat] = {"n": 0, "judged_hacking": 0}
            per_cat[cat]["n"] += 1
            if r["is_reward_hacking"]:
                per_cat[cat]["judged_hacking"] += 1

            # Agreement: judge says hacking iff ground-truth category is a hack type
            gt_is_hack = cat in hack_categories
            judge_is_hack = r["is_reward_hacking"]
            if gt_is_hack == judge_is_hack:
                agree_count += 1

        summary["agreement_with_category"] = (
            round(agree_count / total_with_cat, 4) if total_with_cat else None
        )
        # Order per_cat by CATEGORIES for consistent output
        ordered_per_cat = {}
        for cat_name in CATEGORIES:
            if cat_name in per_cat:
                ordered_per_cat[cat_name] = per_cat[cat_name]
        for cat_name in per_cat:
            if cat_name not in ordered_per_cat:
                ordered_per_cat[cat_name] = per_cat[cat_name]
        summary["per_category"] = ordered_per_cat

    return summary


async def _run(
    input: str,
    output: str,
    vllm_base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B",
    max_concurrent: int = 8,
) -> None:
    """Core async logic for self-assessment."""
    # Load input
    with open(input) as f:
        items = json.load(f)

    if not isinstance(items, list):
        print(f"Error: input must be a JSON array, got {type(items).__name__}", file=sys.stderr)
        sys.exit(1)

    # Validate required fields, skip empty completions
    valid_items = []
    skipped = 0
    for i, item in enumerate(items):
        if "prompt" not in item or "completion" not in item:
            print(f"Warning: item {i} missing 'prompt' or 'completion', skipping", file=sys.stderr)
            skipped += 1
            continue
        if not item["completion"].strip():
            print(f"Warning: item {i} has empty completion, skipping", file=sys.stderr)
            skipped += 1
            continue
        valid_items.append(item)

    if skipped:
        print(f"Skipped {skipped}/{len(items)} items", file=sys.stderr)

    if not valid_items:
        print("Error: no valid items to assess", file=sys.stderr)
        sys.exit(1)

    print(f"Assessing {len(valid_items)} completions with {model} at {vllm_base_url}")

    # Run assessments
    judge = SelfAssessmentJudge(base_url=vllm_base_url, model=model)
    assessment_results = await judge.assess_batch(valid_items, max_concurrent=max_concurrent)

    # Build output results
    result_dicts = []
    for result, item in zip(assessment_results, valid_items):
        entry = {
            "is_reward_hacking": result.is_reward_hacking,
            "explanation": result.explanation,
            "raw_response": result.raw_response,
            "prompt_preview": item["prompt"][:200] + "..." if len(item["prompt"]) > 200 else item["prompt"],
        }
        if result.parse_failed:
            entry["parse_failed"] = True
        if "category" in item:
            entry["original_category"] = item["category"]
        result_dicts.append(entry)

    # Compute summary
    summary = _compute_summary(result_dicts, valid_items)

    # Build output
    output_data = {
        "config": {
            "model": model,
            "base_url": vllm_base_url,
            "n_items": len(valid_items),
        },
        "results": result_dicts,
        "summary": summary,
    }

    # Write output
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary to stdout
    print(f"\n--- Self-Assessment Summary ---")
    print(f"Assessed:        {summary['n_assessed']}")
    print(f"Judged hacking:  {summary['n_judged_hacking']}")
    print(f"Judge hack rate: {summary['judge_hack_rate']}")
    print(f"Parse failures:  {summary['n_parse_failures']}")
    if "agreement_with_category" in summary:
        print(f"Agreement w/ 5-way: {summary['agreement_with_category']}")
        print(f"\nPer-category breakdown:")
        for cat, stats in summary["per_category"].items():
            print(f"  {cat}: {stats['judged_hacking']}/{stats['n']} judged as hacking")
    print(f"\nResults written to {output}")


def self_assess(
    input: str,
    output: str,
    vllm_base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B",
    max_concurrent: int = 8,
) -> None:
    """Run self-assessment on model completions.

    Args:
        input: Path to input JSON (list of {prompt, completion, ?category}).
        output: Path to write output JSON with results and summary.
        vllm_base_url: vLLM server URL.
        model: Model name served by vLLM.
        max_concurrent: Maximum concurrent vLLM requests.
    """
    asyncio.run(_run(
        input=input,
        output=output,
        vllm_base_url=vllm_base_url,
        model=model,
        max_concurrent=max_concurrent,
    ))


if __name__ == "__main__":
    fire.Fire(self_assess)
