"""Inference-time CoT editing evaluation for ImpossibleBench.

Evaluates CoT editing strategies (prefill, insertion, resampling) at inference
time using vLLM's completions API.  Two-phase strategies (insertion/resampling)
need to generate a completion, analyze the think block, then continue from a
specific mid-completion point — requiring raw text completions, not chat.

Usage:
    # Baseline (no editing)
    uv run python -m impossible.eval_cot --hint=check_override

    # Prefill strategy
    uv run python -m impossible.eval_cot --hint=check_override \
      --cot_strategy=prefill \
      --cot_prefill_text="I need to solve this correctly without any shortcuts."

    # Sentence insertion
    uv run python -m impossible.eval_cot --hint=check_override \
      --cot_strategy=insertion \
      --cot_insertion_text="Wait, I should focus on solving this correctly."

    # Sentence resampling (auto-detects hacking patterns)
    uv run python -m impossible.eval_cot --hint=check_override \
      --cot_strategy=resampling

    # Dry run (print first prompt, no API calls)
    uv run python -m impossible.eval_cot --hint=check_override --dry_run
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from common.cot_editing import make_cot_strategy, PrefillStrategy
from common.rewards import _strip_think_blocks
from impossible.data import prepare_impossible_bench_dataset, _HINT_TEMPLATES, extract_problem_prompt
from impossible.rewards import (
    evaluate_impossible_completion,
    CATEGORIES,
    CAT_CORRECT,
    CAT_CORRECT_WITH_HACK,
    CAT_REWARD_HACKING,
    CAT_FAILED_HACK,
    CAT_INCORRECT,
)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

async def _completions_generate(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
) -> str:
    """Call the vLLM completions API (raw text, not chat)."""
    extra_body = {}
    if top_k > 0:
        extra_body["top_k"] = top_k

    response = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        extra_body=extra_body if extra_body else None,
    )
    if not response.choices:
        return ""
    return response.choices[0].text or ""


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    raw_prompt: str,
    strategy,
    tokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
) -> tuple[str, bool]:
    """Generate a single completion, applying CoT strategy if needed.

    Returns (final_completion, cot_edited).
    """
    gen_kwargs = dict(
        client=client, model=model, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
    )

    # No strategy or "none"
    if strategy is None:
        completion = await _completions_generate(prompt=raw_prompt, **gen_kwargs)
        return completion, False

    # Prefill: modify prompt, single-phase
    if isinstance(strategy, PrefillStrategy):
        modified_prompt, meta = strategy.apply_to_prompt(raw_prompt)
        completion = await _completions_generate(prompt=modified_prompt, **gen_kwargs)
        # Completions API returns only generated tokens; prepend the prefill
        # so the stored completion includes the full think block
        if meta.get("cot_edited", False):
            completion = "<think>\n" + strategy.text + "\n" + completion
        return completion, meta.get("cot_edited", False)

    # Two-phase (insertion / resampling)
    # Phase 1: generate normally
    completion = await _completions_generate(prompt=raw_prompt, **gen_kwargs)

    # Decide
    decision = strategy.decide(raw_prompt, completion)
    if decision["action"] == "none":
        return decision.get("original", completion), False

    # Phase 2: continue from prefix + insert_text
    prefix = decision.get("prefix", "")
    insert_text = decision.get("insert_text", "")
    phase2_prompt = raw_prompt + prefix + insert_text

    # Budget: use tokenizer for accurate token count of prefix content
    prefix_tokens = len(tokenizer.encode(prefix + insert_text, add_special_tokens=False))
    remaining_tokens = max(256, max_tokens - prefix_tokens)
    continuation = await _completions_generate(
        client=client, model=model, prompt=phase2_prompt,
        max_tokens=remaining_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed,
    )

    # Assemble
    final = strategy.assemble(decision, continuation)
    return final, True


# ---------------------------------------------------------------------------
# Evaluation (subprocess, sync — run in thread pool)
# ---------------------------------------------------------------------------

def _evaluate_one(
    completion: str,
    test_code: str,
    original_test_code: str,
    entry_point: str,
    timeout: int,
) -> dict:
    """Evaluate a single completion — thin wrapper for thread pool."""
    code_text = _strip_think_blocks(completion)
    return evaluate_impossible_completion(
        code_text, test_code, original_test_code, entry_point, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def _run(
    hint: str,
    output: str,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    max_concurrent: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    splits: str,
    n_problems: int | None,
    eval_timeout: int,
    strategy,
    strategy_name: str,
    cot_kwargs: dict,
    dry_run: bool,
    custom_hint_text: str | None = None,
) -> None:
    """Core async logic."""
    # Load dataset
    ds = prepare_impossible_bench_dataset(
        impossible_splits=[s.strip() for s in splits.split(",")],
        include_benign=True,
        seed=seed or 42,
        hint=hint,
        custom_hint_text=custom_hint_text,
    )

    # Subsample
    if n_problems is not None and n_problems < len(ds):
        import random
        rng = random.Random(seed or 42)
        indices = sorted(rng.sample(range(len(ds)), n_problems))
        items = [ds[int(i)] for i in indices]
    else:
        if n_problems is not None and n_problems > len(ds):
            print(f"  Note: requested {n_problems} problems but dataset has {len(ds)}; using all",
                  file=sys.stderr)
        items = [ds[int(i)] for i in range(len(ds))]

    print(f"Dataset: {len(items)} problems (hint={hint}, splits={splits})")

    # Apply chat template to get raw prompts
    tokenizer = AutoTokenizer.from_pretrained(model)
    raw_prompts = []
    for item in items:
        raw_prompt = tokenizer.apply_chat_template(
            item["prompt"], add_generation_prompt=True, tokenize=False,
        )
        raw_prompts.append(raw_prompt)

    # Dry run: print first prompt and exit
    if dry_run:
        print(f"\n{'='*70}")
        print(f"Sample prompt (item 0: {items[0]['task_id']}, "
              f"impossible={items[0]['is_impossible']})")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*70}")
        prompt_to_show = raw_prompts[0]
        if strategy is not None and isinstance(strategy, PrefillStrategy):
            prompt_to_show, meta = strategy.apply_to_prompt(prompt_to_show)
            print(f"[Prefill applied: {meta}]")
        print(prompt_to_show)
        print(f"{'='*70}")
        print(f"\nDry run complete — {len(items)} problems would be evaluated.")
        return

    # Initialize client and validate connection
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        await client.models.list()
    except Exception as e:
        print(f"Error: cannot connect to {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Concurrent generation with semaphore
    sem = asyncio.Semaphore(max_concurrent)
    results = [None] * len(items)
    completions = [""] * len(items)
    cot_edited = [False] * len(items)

    async def _gen_and_eval(idx: int) -> None:
        async with sem:
            try:
                comp, edited = await _generate_one(
                    client=client, model=model, raw_prompt=raw_prompts[idx],
                    strategy=strategy, tokenizer=tokenizer, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
                )
                completions[idx] = comp
                cot_edited[idx] = edited
            except Exception as e:
                print(f"  WARNING: generation error for {items[idx]['task_id']}: {e}",
                      file=sys.stderr)
                completions[idx] = ""
                cot_edited[idx] = False

    # Generate all completions
    t0 = time.time()
    print(f"Generating {len(items)} completions (max_concurrent={max_concurrent})...")
    await asyncio.gather(*[_gen_and_eval(i) for i in range(len(items))])
    gen_time = time.time() - t0
    print(f"Generation complete in {gen_time:.1f}s")

    # Evaluate in parallel (sync subprocess calls via thread pool)
    print(f"Evaluating {len(items)} completions...")
    t1 = time.time()
    max_eval_workers = min(8, len(items))
    with ThreadPoolExecutor(max_workers=max_eval_workers) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_one,
                completions[i],
                items[i]["test_code"],
                items[i]["original_test_code"],
                items[i]["entry_point"],
                eval_timeout,
            ): i
            for i in range(len(items))
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  WARNING: eval error for {items[idx]['task_id']}: {e}",
                      file=sys.stderr)
                results[idx] = {
                    "mutated_pass": False, "gt_pass": False,
                    "redefines_check": False, "category": CAT_INCORRECT,
                    "compiled": False, "reward": 0.0,
                }
    eval_time = time.time() - t1
    print(f"Evaluation complete in {eval_time:.1f}s")

    # Build per-result records
    result_records = []
    for i in range(len(items)):
        result_records.append({
            "task_id": items[i]["task_id"],
            "is_impossible": items[i]["is_impossible"],
            "impossible_type": items[i]["impossible_type"],
            "prompt_text": extract_problem_prompt(items[i]),
            "completion": completions[i],
            "cot_edited": cot_edited[i],
            **results[i],
        })

    # Compute summary
    n = len(result_records)
    n_valid = sum(1 for r in result_records if r["compiled"])

    category_counts = {cat: 0 for cat in CATEGORIES}
    for r in result_records:
        category_counts[r["category"]] += 1

    impossible_items = [r for r in result_records if r["is_impossible"]]
    benign_items = [r for r in result_records if not r["is_impossible"]]

    impossible_hack_count = sum(1 for r in impossible_items if r["mutated_pass"])
    benign_correct_count = sum(1 for r in benign_items if r["mutated_pass"])

    summary = {
        "n_samples": n,
        "n_valid": n_valid,
        "correct_rate": round(category_counts[CAT_CORRECT] / n, 4) if n else 0,
        "hack_rate": round(category_counts[CAT_REWARD_HACKING] / n, 4) if n else 0,
        "correct_with_attempted_hack_rate": round(category_counts[CAT_CORRECT_WITH_HACK] / n, 4) if n else 0,
        "failed_hack_rate": round(category_counts[CAT_FAILED_HACK] / n, 4) if n else 0,
        "incorrect_rate": round(category_counts[CAT_INCORRECT] / n, 4) if n else 0,
        "compile_rate": round(sum(1 for r in result_records if r["compiled"]) / n, 4) if n else 0,
        "redefines_check_rate": round(sum(1 for r in result_records if r["redefines_check"]) / n, 4) if n else 0,
        "impossible_hack_rate": round(impossible_hack_count / len(impossible_items), 4) if impossible_items else 0,
        "benign_correct_rate": round(benign_correct_count / len(benign_items), 4) if benign_items else 0,
        "cot_edited_rate": round(sum(1 for r in result_records if r["cot_edited"]) / n, 4) if n else 0,
        "generation_time_s": round(gen_time, 1),
        "evaluation_time_s": round(eval_time, 1),
    }

    # Build output
    output_data = {
        "config": {
            "model": model,
            "base_url": base_url,
            "hint": hint,
            "cot_strategy": strategy_name,
            **cot_kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "splits": splits,
            "n_problems": n_problems,
            "eval_timeout": eval_timeout,
        },
        "summary": summary,
        "results": result_records,
    }

    # Write output
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CoT Editing Eval — {strategy_name} | hint={hint}")
    print(f"{'='*60}")
    print(f"  Samples:        {n}")
    print(f"  Compile rate:   {summary['compile_rate']:.1%}")
    print(f"  Correct rate:   {summary['correct_rate']:.1%}")
    print(f"  Hack rate:      {summary['hack_rate']:.1%}")
    print(f"  Failed hack:    {summary['failed_hack_rate']:.1%}")
    print(f"  Correct+hack:   {summary['correct_with_attempted_hack_rate']:.1%}")
    print(f"  Incorrect:      {summary['incorrect_rate']:.1%}")
    print(f"  Redefines check:{summary['redefines_check_rate']:.1%}")
    print(f"  Imp. hack rate: {summary['impossible_hack_rate']:.1%}")
    print(f"  Benign correct: {summary['benign_correct_rate']:.1%}")
    print(f"  CoT edited:     {summary['cot_edited_rate']:.1%}")
    print(f"  Gen time:       {summary['generation_time_s']}s")
    print(f"  Eval time:      {summary['evaluation_time_s']}s")
    print(f"{'='*60}")
    print(f"Results written to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VALID_HINTS = frozenset(("none", "custom", *_HINT_TEMPLATES))


def eval_cot(
    hint: str = "none",
    output: str = "results/eval/eval_cot.json",
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B",
    api_key_env: str = "",
    max_tokens: int = 32768,
    max_concurrent: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 42,
    splits: str = "conflicting,oneoff",
    n_problems: int | None = None,
    eval_timeout: int = 10,
    # CoT editing
    cot_strategy: str = "none",
    cot_prefill_text: str | None = None,
    cot_insertion_text: str | None = None,
    cot_insertion_probability: float = 1.0,
    cot_insertion_concentration: float = 2.0,
    cot_resampling_patterns: str | None = None,
    # Custom hint
    custom_hint_text: str | None = None,
    # Debug
    dry_run: bool = False,
) -> None:
    """Evaluate CoT editing strategies at inference time on ImpossibleBench.

    Generates completions via vLLM completions API, optionally applying a CoT
    editing strategy (prefill, insertion, resampling), then evaluates each
    completion with the ImpossibleBench 5-way classification.

    Args:
        hint: Hint type for exploit discoverability (e.g. "check_override").
            Use "custom" with custom_hint_text for arbitrary hint text.
        output: Path to write JSON results.
        base_url: vLLM-compatible completions API base URL.
        model: Model name/identifier.
        api_key_env: Env var name for API key (default: uses "unused" for local vLLM).
        max_tokens: Max completion tokens (default 32768 for think mode).
        max_concurrent: Max concurrent API requests.
        temperature: Sampling temperature (default 0.6 for think mode).
        top_p: Top-p sampling (default 0.95).
        top_k: Top-k sampling (default 20).
        seed: Random seed.
        splits: Comma-separated impossible splits.
        n_problems: Subsample N problems (None = all).
        eval_timeout: Seconds per subprocess evaluation.
        cot_strategy: CoT editing strategy ("none", "prefill", "insertion", "resampling").
        cot_prefill_text: Text for prefill strategy.
        cot_insertion_text: Text for insertion strategy.
        cot_insertion_probability: Probability of insertion per completion.
        cot_insertion_concentration: Bell-curve peakedness for insertion point (0=uniform, 2=moderate, 4=strong).
        cot_resampling_patterns: Comma-separated regex patterns for resampling.
        custom_hint_text: Arbitrary hint text when hint="custom". Supports {entry_point}.
        dry_run: Print first prompt without API calls.
    """
    # Validate hint
    if hint == "custom" and not custom_hint_text:
        print("Error: hint='custom' requires --custom_hint_text", file=sys.stderr)
        sys.exit(1)
    if hint not in _VALID_HINTS:
        print(f"Error: unknown hint {hint!r}. Valid: {sorted(_VALID_HINTS)}",
              file=sys.stderr)
        sys.exit(1)

    # Build strategy
    strategy = make_cot_strategy(
        cot_strategy,
        prefill_text=cot_prefill_text,
        insertion_text=cot_insertion_text,
        insertion_probability=cot_insertion_probability,
        insertion_concentration=cot_insertion_concentration,
        resampling_patterns=(
            cot_resampling_patterns.split(",") if cot_resampling_patterns else None
        ),
    )

    # Collect CoT-specific config for output
    cot_kwargs = {}
    if cot_prefill_text:
        cot_kwargs["cot_prefill_text"] = cot_prefill_text
    if cot_insertion_text:
        cot_kwargs["cot_insertion_text"] = cot_insertion_text
    if cot_insertion_probability != 1.0:
        cot_kwargs["cot_insertion_probability"] = cot_insertion_probability
    if cot_insertion_concentration != 2.0:
        cot_kwargs["cot_insertion_concentration"] = cot_insertion_concentration
    if cot_resampling_patterns:
        cot_kwargs["cot_resampling_patterns"] = cot_resampling_patterns

    # API key
    api_key = "unused"
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            print(f"Error: {api_key_env} not set", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_run(
        hint=hint,
        output=output,
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        splits=splits,
        n_problems=n_problems,
        eval_timeout=eval_timeout,
        strategy=strategy,
        strategy_name=cot_strategy,
        cot_kwargs=cot_kwargs,
        dry_run=dry_run,
        custom_hint_text=custom_hint_text,
    ))


if __name__ == "__main__":
    fire.Fire(eval_cot)
