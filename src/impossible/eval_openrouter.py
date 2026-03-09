"""Zero-shot OpenRouter CoT prefill evaluation for ImpossibleBench.

Tests whether CoT prefilling can prevent reward hacking in zero-shot settings
using OpenRouter's completions API for full control over <think> block injection.

Usage:
    # Baseline (no CoT editing) — find if model reward hacks
    uv run python -m impossible.eval_openrouter \
        --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 \
        --hint=check_override --n_problems=10

    # Prefill strategy
    uv run python -m impossible.eval_openrouter \
        --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 \
        --hint=check_override --n_problems=10 \
        --cot_strategy=prefill \
        --cot_prefill_text="I need to solve this correctly without shortcuts."

    # Sentence insertion
    uv run python -m impossible.eval_openrouter \
        --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 \
        --hint=check_override --n_problems=10 \
        --cot_strategy=insertion \
        --cot_insertion_text="Wait, I should focus on solving this correctly."

    # Sentence resampling (auto-detects hacking patterns)
    uv run python -m impossible.eval_openrouter \
        --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 \
        --hint=check_override --n_problems=10 \
        --cot_strategy=resampling

    # Dry run — inspect prompt format
    uv run python -m impossible.eval_openrouter --dry_run --n_problems=1
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
from impossible.data import prepare_impossible_bench_dataset, extract_problem_prompt
from impossible.hints import HINT_TEMPLATES
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
) -> str:
    """Call the OpenRouter completions API (raw text, not chat).

    Unlike vLLM, OpenRouter does not support top_k or seed on completions.
    """
    response = await client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if not response.choices:
        return ""
    return response.choices[0].text or ""


def _apply_prefill(raw_prompt: str, prefill_text: str) -> str:
    """Model-agnostic prefill: append prefill text after <think> in raw prompt.

    Handles two cases:
    - Template already includes <think> at the end (e.g. Kimi K2.5)
    - Template doesn't include <think> (e.g. Qwen3) — we add it
    """
    stripped = raw_prompt.rstrip()
    if stripped.endswith("<think>"):
        return raw_prompt + prefill_text + "\n"
    else:
        return raw_prompt + "<think>\n" + prefill_text + "\n"


async def _generate_one(
    client: AsyncOpenAI,
    model: str,
    raw_prompt: str,
    strategy,
    tokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, bool]:
    """Generate a single completion, applying CoT strategy if needed.

    Returns (final_completion, cot_edited).
    """
    gen_kwargs = dict(
        client=client, model=model, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
    )

    # No strategy
    if strategy is None:
        completion = await _completions_generate(prompt=raw_prompt, **gen_kwargs)
        return completion, False

    # Prefill: model-agnostic approach (not PrefillStrategy.apply_to_prompt which is Qwen3-specific)
    if isinstance(strategy, PrefillStrategy):
        prefilled_prompt = _apply_prefill(raw_prompt, strategy.text)
        completion = await _completions_generate(prompt=prefilled_prompt, **gen_kwargs)
        # Prepend <think> + prefill text so the stored completion includes the
        # full think block (needed for _strip_think_blocks to find it later).
        # Both cases need <think>\n — when the template already has <think>,
        # it's part of the prompt (not echoed by the API).
        full_completion = "<think>\n" + strategy.text + "\n" + completion
        return full_completion, True

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
        max_tokens=remaining_tokens, temperature=temperature, top_p=top_p,
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
    hack_vector: str,
) -> dict:
    """Evaluate a single completion — thin wrapper for thread pool.

    Passes the full completion text (after stripping <think> blocks) to
    evaluate_impossible_completion, which uses its own code parser to
    extract fenced code blocks.
    """
    code_text = _strip_think_blocks(completion)
    return evaluate_impossible_completion(
        code_text, test_code, original_test_code, entry_point,
        timeout=timeout, hack_vector=hack_vector,
    )


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _estimate_cost(items: list, raw_prompts: list[str], max_tokens: int, model: str) -> None:
    """Print estimated token counts and cost before starting."""
    total_input_chars = sum(len(p) for p in raw_prompts)
    # Rough estimate: 1 token ≈ 3.5 chars for code-heavy content
    est_input_tokens = total_input_chars / 3.5
    est_output_tokens = len(items) * max_tokens * 0.5  # assume ~50% of budget used

    print(f"Cost estimate for {model}:")
    print(f"  Est. input tokens:  {est_input_tokens:,.0f}")
    print(f"  Est. output tokens: {est_output_tokens:,.0f}")
    print(f"  (Actual cost depends on model pricing and completion length)")


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------

async def _run(
    hint: str,
    output: str,
    base_url: str,
    model: str,
    tokenizer_name: str,
    api_key: str,
    max_tokens: int,
    max_concurrent: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    splits: str,
    n_problems: int | None,
    eval_timeout: int,
    hack_vector: str,
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
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
            prompt_to_show = _apply_prefill(prompt_to_show, strategy.text)
            print("[Prefill applied]")
        print(prompt_to_show)
        print(f"{'='*70}")
        print(f"\nDry run complete — {len(items)} problems would be evaluated.")
        return

    # Print cost estimate
    _estimate_cost(items, raw_prompts, max_tokens, model)

    # Initialize client with OpenRouter headers
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=3,
        default_headers={
            "X-Title": "cot-exploration",
            "HTTP-Referer": "https://github.com/cot-exploration",
            "X-Safety-Identifier": "mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:cot-exploration",
        },
    )

    # Concurrent generation with semaphore
    sem = asyncio.Semaphore(max_concurrent)
    results = [None] * len(items)
    completions = [""] * len(items)
    cot_edited = [False] * len(items)
    _consecutive_failures = 0
    _MAX_CONSECUTIVE_FAILURES = 5
    _abort = False

    async def _gen_and_eval(idx: int) -> None:
        nonlocal _consecutive_failures, _abort
        if _abort:
            return
        async with sem:
            if _abort:
                return
            try:
                comp, edited = await _generate_one(
                    client=client, model=model, raw_prompt=raw_prompts[idx],
                    strategy=strategy, tokenizer=tokenizer, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                )
                completions[idx] = comp
                cot_edited[idx] = edited
                _consecutive_failures = 0
            except Exception as e:
                _consecutive_failures += 1
                print(f"  WARNING: generation error for {items[idx]['task_id']}: {e}",
                      file=sys.stderr)
                completions[idx] = ""
                cot_edited[idx] = False
                if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"\n  ABORTING: {_consecutive_failures} consecutive failures "
                          f"(credits exhausted? API down?)", file=sys.stderr)
                    _abort = True

    # Generate all completions
    t0 = time.time()
    print(f"Generating {len(items)} completions (max_concurrent={max_concurrent})...")
    await asyncio.gather(*[_gen_and_eval(i) for i in range(len(items))])
    gen_time = time.time() - t0
    n_generated = sum(1 for c in completions if c)
    print(f"Generation complete in {gen_time:.1f}s ({n_generated}/{len(items)} succeeded)")

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
                hack_vector,
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

    category_counts = {cat: 0 for cat in CATEGORIES}
    for r in result_records:
        category_counts[r["category"]] += 1

    impossible_items = [r for r in result_records if r["is_impossible"]]
    benign_items = [r for r in result_records if not r["is_impossible"]]

    impossible_hack_count = sum(1 for r in impossible_items if r["mutated_pass"])
    benign_correct_count = sum(1 for r in benign_items if r["mutated_pass"])

    summary = {
        "n_samples": n,
        "n_valid": sum(1 for r in result_records if r["compiled"]),
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
            "tokenizer": tokenizer_name,
            "base_url": base_url,
            "hint": hint,
            "hack_vector": hack_vector,
            "cot_strategy": strategy_name,
            **cot_kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
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
    print(f"OpenRouter Eval — {strategy_name} | hint={hint} | model={model}")
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

_VALID_HINTS = frozenset(("none", "custom", *HINT_TEMPLATES))


def eval_openrouter(
    model: str = "moonshotai/kimi-k2.5",
    tokenizer: str = "moonshotai/Kimi-K2.5",
    hint: str = "check_override",
    output: str = "results/eval/eval_openrouter.json",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env: str = "OPENROUTER_API_KEY",
    max_tokens: int = 16384,
    max_concurrent: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 42,
    splits: str = "conflicting,oneoff",
    n_problems: int | None = None,
    eval_timeout: int = 10,
    hack_vector: str = "check_redef",
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
    """Evaluate CoT editing strategies via OpenRouter on ImpossibleBench.

    Uses OpenRouter's completions API (raw text) for full control over prompt
    including <think> block injection. Supports all three CoT editing strategies.

    Args:
        model: OpenRouter model name (e.g. "moonshotai/kimi-k2.5").
        tokenizer: HuggingFace tokenizer name (e.g. "moonshotai/Kimi-K2.5").
        hint: Hint type for exploit discoverability.
        output: Path to write JSON results.
        base_url: OpenRouter API base URL.
        api_key_env: Env var name for API key.
        max_tokens: Max completion tokens.
        max_concurrent: Max concurrent API requests.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        seed: Random seed (for dataset subsampling; not passed to API).
        splits: Comma-separated impossible splits.
        n_problems: Subsample N problems (None = all).
        eval_timeout: Seconds per subprocess evaluation.
        hack_vector: Detection mode ("check_redef" or "sys_exit").
        cot_strategy: CoT editing strategy ("none", "prefill", "insertion", "resampling").
        cot_prefill_text: Text for prefill strategy.
        cot_insertion_text: Text for insertion strategy.
        cot_insertion_probability: Probability of insertion per completion.
        cot_insertion_concentration: Bell-curve peakedness for insertion point.
        cot_resampling_patterns: Comma-separated regex patterns for resampling.
        custom_hint_text: Arbitrary hint text when hint="custom".
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

    # Validate hack_vector
    from impossible.rewards import HACK_VECTORS
    if hack_vector not in HACK_VECTORS:
        print(f"Error: unknown hack_vector {hack_vector!r}. Valid: {HACK_VECTORS}",
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
    api_key = os.environ.get(api_key_env, "")
    if not api_key and not dry_run:
        print(f"Error: {api_key_env} not set", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        api_key = "unused"  # dry_run doesn't need a real key

    asyncio.run(_run(
        hint=hint,
        output=output,
        base_url=base_url,
        model=model,
        tokenizer_name=tokenizer,
        api_key=api_key,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        splits=splits,
        n_problems=n_problems,
        eval_timeout=eval_timeout,
        hack_vector=hack_vector,
        strategy=strategy,
        strategy_name=cot_strategy,
        cot_kwargs=cot_kwargs,
        dry_run=dry_run,
        custom_hint_text=custom_hint_text,
    ))


if __name__ == "__main__":
    fire.Fire(eval_openrouter)
