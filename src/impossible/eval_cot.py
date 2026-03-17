"""Inference-time CoT editing evaluation for ImpossibleBench.

Evaluates CoT editing strategies (prefill, insertion, resampling, redirect) at inference
time using vLLM's completions API.  Two-phase strategies (insertion/resampling)
need to generate a completion, analyze the think block, then continue from a
specific mid-completion point -- requiring raw text completions, not chat.

Supports two prompt formats:
- "think" (default): Qwen3-style <think>...</think> blocks via chat template
- "harmony": OpenAI gpt-oss Harmony format via openai_harmony library
  (uses token IDs for special tokens, analysis/final channel structure)

Usage:
    # Baseline (no editing)
    uv run python -m impossible.eval_cot --hint=check_override

    # Prefill strategy
    uv run python -m impossible.eval_cot --hint=check_override \\
      --cot_strategy=prefill \\
      --cot_prefill_text="I need to solve this correctly without any shortcuts."

    # Harmony format (gpt-oss)
    uv run python -m impossible.eval_cot \\
      --model=openai/gpt-oss-120b --tokenizer=openai/gpt-oss-120b \\
      --format=harmony --hint=check_override --hack_vector=sys_exit \\
      --exemplar_file=src/impossible/sysexit_exemplars.json \\
      --impossible_only --dry_run

    # Dry run (print first prompt, no API calls)
    uv run python -m impossible.eval_cot --hint=check_override --dry_run
"""

import asyncio
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fire
from openai import AsyncOpenAI

from common.cot_editing import (
    make_cot_strategy, PrefillStrategy, SentenceInsertionStrategy,
    find_sentence_boundaries, _middle_biased_weights,
)
from common.rewards import _strip_think_blocks
from impossible.data import prepare_impossible_bench_dataset, extract_problem_prompt
from impossible.hints import HINT_TEMPLATES
from impossible.rewards import (
    evaluate_impossible_completion,
    HACK_VECTORS,
    CATEGORIES,
    CAT_CORRECT,
    CAT_CORRECT_WITH_HACK,
    CAT_REWARD_HACKING,
    CAT_FAILED_HACK,
    CAT_INCORRECT,
)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_HARMONY_MODEL_PATTERNS = ("gpt-oss",)


def _detect_format(model: str, fmt: str) -> str:
    """Resolve 'auto' format based on model name."""
    if fmt != "auto":
        return fmt
    for pat in _HARMONY_MODEL_PATTERNS:
        if pat in model.lower():
            return "harmony"
    return "think"


# ---------------------------------------------------------------------------
# Harmony helpers
# ---------------------------------------------------------------------------

# Regex to extract final channel content from Harmony output
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|channel\|>|$)",
    re.DOTALL,
)
# Regex to extract analysis channel content
_HARMONY_ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|channel\|>|<\|end\|>|$)",
    re.DOTALL,
)


def _strip_harmony_analysis(text: str) -> str:
    """Extract final channel content from Harmony format output.

    If the output has explicit channel markers, extracts the ``final``
    channel content.  Falls back to the full text if no channel markers found
    (e.g. if model didn't use channels).
    """
    m = _HARMONY_FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    # No channel markers -- return full text (may happen with short outputs)
    return text.strip()


def _build_harmony_prompt_ids(
    system_text: str,
    messages: list[dict],
    exemplar_turns: list[dict],
) -> list[int]:
    """Build Harmony-format token IDs for a prompt.

    Uses openai_harmony (tiktoken-based) to correctly encode special tokens
    including the system message as ``<|start|>system<|message|>...<|end|>``.
    The openai_harmony library's render_conversation omits system messages from
    the token stream, but the actual Harmony format includes them.
    """
    from openai_harmony import load_harmony_encoding
    tok = load_harmony_encoding("HarmonyGptOss")
    allowed = {"<|start|>", "<|end|>", "<|message|>", "<|channel|>"}

    # Build the full prompt text with Harmony special tokens
    parts = []

    # System message
    if system_text:
        parts.append(f"<|start|>system<|message|>{system_text}<|end|>")

    # Exemplar turns (between system and real user message)
    for turn in exemplar_turns:
        role = turn["role"]
        content = turn["content"]
        if role == "assistant":
            if "analysis_content" in turn:
                # Multichannel: analysis + final channels (matches model's natural output)
                parts.append(f"<|start|>assistant<|channel|>analysis<|message|>{turn['analysis_content']}<|end|>")
                parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")
            else:
                # Single-channel: no channel markers (legacy format)
                parts.append(f"<|start|>assistant<|message|>{content}<|end|>")
        else:
            parts.append(f"<|start|>{role}<|message|>{content}<|end|>")

    # Real messages (skip system, already added above)
    for msg in messages:
        if msg["role"] == "system":
            continue
        parts.append(f"<|start|>{msg['role']}<|message|>{msg['content']}<|end|>")

    # Add assistant generation prompt
    parts.append("<|start|>assistant")

    full_text = "".join(parts)
    return tok.encode(full_text, allowed_special=allowed)


def _build_harmony_prefill_ids(prefill_text: str) -> list[int]:
    """Build token IDs for Harmony analysis channel prefill.

    Returns: [<|channel|>, analysis, <|message|>, ...prefill_text...]
    """
    from openai_harmony import load_harmony_encoding
    tok = load_harmony_encoding("HarmonyGptOss")
    allowed = {"<|channel|>", "<|message|>"}
    return tok.encode(
        f"<|channel|>analysis<|message|>{prefill_text}",
        allowed_special=allowed,
    )


def _decode_harmony_tokens(token_ids: list[int]) -> str:
    """Decode Harmony token IDs to text."""
    from openai_harmony import load_harmony_encoding
    tok = load_harmony_encoding("HarmonyGptOss")
    return tok.decode(token_ids)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

async def _completions_generate(
    client: AsyncOpenAI,
    model: str,
    prompt,  # str or list[int] (token IDs)
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    reconstruct_special_tokens: bool = False,
) -> str:
    """Call the vLLM completions API (raw text, not chat).

    Prompt can be a string (think format) or list[int] (harmony token IDs).

    When *reconstruct_special_tokens* is True, requests logprobs=1 and
    reconstructs text from logprob tokens so that special tokens (e.g.
    Harmony ``<|channel|>``, ``<|message|>``) are preserved.  Without this,
    vLLM strips special tokens from the text field.
    """
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
        logprobs=1 if reconstruct_special_tokens else None,
        extra_body=extra_body if extra_body else None,
    )
    if not response.choices:
        return ""

    choice = response.choices[0]
    if reconstruct_special_tokens and choice.logprobs and choice.logprobs.tokens:
        return "".join(choice.logprobs.tokens)
    return choice.text or ""


async def _generate_one_think(
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
    """Generate a single completion in think format (Qwen3-style).

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

    # Iterative (redirect) — loops until clean or max_iterations
    if getattr(strategy, 'needs_iterative', False):
        return await _generate_one_think_iterative(
            client=client, model=model, raw_prompt=raw_prompt,
            strategy=strategy, tokenizer=tokenizer,
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed,
        )

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


async def _generate_one_think_iterative(
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
    """Generate with iterative redirect: detect keywords, replace, regenerate, loop.

    Returns (final_completion, cot_edited).
    """
    gen_kwargs = dict(
        client=client, model=model, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
    )

    # Phase 1: generate normally
    current_completion = await _completions_generate(prompt=raw_prompt, **gen_kwargs)
    was_edited = False

    for _ in range(strategy.max_iterations):
        decision = strategy.decide(raw_prompt, current_completion)
        if decision["action"] == "none":
            break

        prefix = decision.get("prefix", "")
        insert_text = decision.get("insert_text", "")
        phase_prompt = raw_prompt + prefix + insert_text

        prefix_tokens = len(tokenizer.encode(prefix + insert_text, add_special_tokens=False))
        remaining_tokens = max(256, max_tokens - prefix_tokens)
        continuation = await _completions_generate(
            client=client, model=model, prompt=phase_prompt,
            max_tokens=remaining_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed,
        )

        current_completion = strategy.assemble(decision, continuation)
        was_edited = True

    return current_completion, was_edited


async def _generate_one_harmony(
    client: AsyncOpenAI,
    model: str,
    prompt_ids: list[int],
    strategy,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    context_limit: int | None = None,
) -> tuple[str, bool]:
    """Generate a single completion in Harmony format.

    Returns (final_completion, cot_edited).
    Supports prefill and insertion strategies.
    """
    import random as _random
    gen_kwargs = dict(
        client=client, model=model, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed,
    )

    # Harmony needs logprobs-based reconstruction to preserve special tokens
    gen_kwargs["reconstruct_special_tokens"] = True

    if isinstance(strategy, PrefillStrategy):
        # Inject analysis channel prefill after the prompt
        prefill_ids = _build_harmony_prefill_ids(strategy.text)
        full_prompt_ids = prompt_ids + prefill_ids
        completion = await _completions_generate(prompt=full_prompt_ids, **gen_kwargs)
        # Prepend the prefill so stored completion has full analysis context
        prefill_marker = f"<|channel|>analysis<|message|>{strategy.text}"
        return prefill_marker + completion, True

    if isinstance(strategy, SentenceInsertionStrategy):
        # Phase 1: generate normally
        completion = await _completions_generate(prompt=prompt_ids, **gen_kwargs)

        # Extract analysis channel
        analysis_match = _HARMONY_ANALYSIS_RE.search(completion)
        if not analysis_match:
            return completion, False  # No analysis channel → no edit

        analysis_text = analysis_match.group(1)

        # Insertion probability check
        if strategy.probability < 1.0 and _random.random() > strategy.probability:
            return completion, False

        # Find sentence boundaries within analysis text
        boundaries = find_sentence_boundaries(analysis_text)
        if not boundaries:
            return completion, False  # No boundaries → no edit

        # Pick boundary (middle-biased)
        weights = _middle_biased_weights(len(boundaries), strategy.concentration)
        boundary = _random.choices(boundaries, weights=weights, k=1)[0]

        # Cut position in full completion text
        cut_pos = analysis_match.start(1) + boundary
        prefix_text = completion[:cut_pos]
        insert_text = strategy.text + " "

        # Re-encode with FULL Harmony special token set
        from openai_harmony import load_harmony_encoding
        tok = load_harmony_encoding("HarmonyGptOss")
        allowed = {"<|start|>", "<|end|>", "<|message|>", "<|channel|>"}
        prefix_ids = tok.encode(prefix_text, allowed_special=allowed)
        insert_ids = tok.encode(insert_text)  # plain text, no special tokens

        # Phase 2 context limit guard
        phase2_prompt_len = len(prompt_ids) + len(prefix_ids) + len(insert_ids)
        if context_limit is not None and phase2_prompt_len + 256 > context_limit:
            return completion, False  # Can't fit phase 2 → return unedited

        # Phase 2: generate continuation
        remaining_tokens = max(256, max_tokens - len(prefix_ids) - len(insert_ids))
        phase2_prompt_ids = prompt_ids + prefix_ids + insert_ids
        continuation = await _completions_generate(
            prompt=phase2_prompt_ids,
            client=client, model=model,
            max_tokens=remaining_tokens,
            temperature=temperature, top_p=top_p,
            top_k=top_k, seed=seed,
            reconstruct_special_tokens=True,
        )

        # Assemble
        return prefix_text + insert_text + continuation, True

    # Iterative (redirect) for Harmony — works on analysis channel text.
    # Uses public find_keyword_sentence() and replacement_text (not decide(),
    # which depends on split_think_block / <think> format).
    if getattr(strategy, 'needs_iterative', False):
        from openai_harmony import load_harmony_encoding
        tok = load_harmony_encoding("HarmonyGptOss")
        allowed = {"<|start|>", "<|end|>", "<|message|>", "<|channel|>"}  # must match _build_harmony_prompt_ids

        completion = await _completions_generate(prompt=prompt_ids, **gen_kwargs)
        was_edited = False

        for _ in range(strategy.max_iterations):
            # Extract analysis channel to scan for keywords
            analysis_match = _HARMONY_ANALYSIS_RE.search(completion)
            if not analysis_match:
                break  # No analysis channel → can't redirect

            analysis_text = analysis_match.group(1)
            match = strategy.find_keyword_sentence(analysis_text)
            if match is None:
                break  # Clean

            sent_start, sent_end = match
            # Cut at start of matched sentence within analysis text
            cut_pos = analysis_match.start(1) + sent_start
            prefix_text = completion[:cut_pos]
            insert_text = strategy.replacement_text + " "

            # Re-encode for vLLM
            prefix_ids = tok.encode(prefix_text, allowed_special=allowed)
            insert_ids = tok.encode(insert_text)

            # Context limit guard
            phase_prompt_len = len(prompt_ids) + len(prefix_ids) + len(insert_ids)
            if context_limit is not None and phase_prompt_len + 256 > context_limit:
                break

            remaining_tokens = max(256, max_tokens - len(prefix_ids) - len(insert_ids))
            phase_prompt_ids = prompt_ids + prefix_ids + insert_ids
            continuation = await _completions_generate(
                prompt=phase_prompt_ids,
                client=client, model=model,
                max_tokens=remaining_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, seed=seed,
                reconstruct_special_tokens=True,
            )

            completion = prefix_text + insert_text + continuation
            was_edited = True

        return completion, was_edited

    # No strategy (baseline)
    completion = await _completions_generate(prompt=prompt_ids, **gen_kwargs)
    return completion, False


# ---------------------------------------------------------------------------
# Output stripping
# ---------------------------------------------------------------------------

def _strip_cot(completion: str, fmt: str) -> str:
    """Strip CoT/reasoning content from completion based on format."""
    if fmt == "harmony":
        return _strip_harmony_analysis(completion)
    return _strip_think_blocks(completion)


# ---------------------------------------------------------------------------
# Evaluation (subprocess, sync -- run in thread pool)
# ---------------------------------------------------------------------------

def _evaluate_one(
    completion: str,
    test_code: str,
    original_test_code: str,
    entry_point: str,
    timeout: int,
    hack_vector: str,
    fmt: str,
) -> dict:
    """Evaluate a single completion -- thin wrapper for thread pool."""
    code_text = _strip_cot(completion, fmt)
    return evaluate_impossible_completion(
        code_text, test_code, original_test_code, entry_point,
        timeout=timeout, hack_vector=hack_vector,
    )


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
    top_k: int,
    seed: int | None,
    splits: str,
    n_problems: int | None,
    eval_timeout: int,
    hack_vector: str,
    strategy,
    strategy_name: str,
    cot_kwargs: dict,
    dry_run: bool,
    fmt: str,
    custom_hint_text: str | None = None,
    exemplar_file: str | None = None,
    impossible_only: bool = False,
    benign_only: bool = False,
    context_limit: int | None = None,
    reasoning_effort: str | None = None,
) -> None:
    """Core async logic."""
    # Load dataset
    imp_splits = [] if benign_only else [s.strip() for s in splits.split(",")]
    ds = prepare_impossible_bench_dataset(
        impossible_splits=imp_splits,
        include_benign=not impossible_only,
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

    mode = "impossible_only" if impossible_only else "benign_only" if benign_only else "mixed"
    print(f"Dataset: {len(items)} problems (hint={hint}, splits={splits}, mode={mode})")

    # Load exemplar turns if provided
    exemplar_turns: list[dict] = []
    if exemplar_file:
        with open(exemplar_file) as f:
            exemplars = json.load(f)
        for ex in exemplars:
            exemplar_turns.append({"role": "user", "content": ex["user_message"]})
            assistant_turn = {"role": "assistant", "content": ex["assistant_response"]}
            if "analysis_content" in ex:
                assistant_turn["analysis_content"] = ex["analysis_content"]
            exemplar_turns.append(assistant_turn)
        # Exclude exemplar problems from eval set to avoid contamination
        exemplar_ids = {ex["task_id"] for ex in exemplars if "task_id" in ex}
        if exemplar_ids:
            before = len(items)
            items = [it for it in items if it["task_id"] not in exemplar_ids]
            excluded = before - len(items)
            if excluded:
                print(f"Excluded {excluded} exemplar problems from eval set: "
                      f"{sorted(exemplar_ids)}")
        print(f"Loaded {len(exemplars)} exemplars from {exemplar_file} "
              f"({len(exemplar_turns)} turns), {len(items)} problems remaining")

    # Build prompts based on format
    tokenizer = None
    raw_prompts: list[str] = []  # think format
    prompt_id_lists: list[list[int]] = []  # harmony format

    if fmt == "harmony":
        # Extract system text from first item's prompt
        system_text = ""
        for msg in items[0]["prompt"]:
            if msg["role"] == "system":
                system_text = msg["content"]
                break

        # Prepend reasoning effort directive for Harmony format
        if reasoning_effort:
            system_text = f"Reasoning: {reasoning_effort}\n\n" + system_text

        for item in items:
            msgs = item["prompt"]
            ids = _build_harmony_prompt_ids(system_text, msgs, exemplar_turns)
            prompt_id_lists.append(ids)

        print(f"Format: harmony (token IDs, {len(prompt_id_lists[0])} tokens for first prompt)")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        for item in items:
            if exemplar_turns:
                msgs = [item["prompt"][0], *exemplar_turns, *item["prompt"][1:]]
            else:
                msgs = item["prompt"]
            raw_prompt = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
            raw_prompts.append(raw_prompt)
        print(f"Format: think (text prompts, {len(raw_prompts[0])} chars for first prompt)")

    # Dry run: print first prompt and exit
    if dry_run:
        print(f"\n{'='*70}")
        print(f"Sample prompt (item 0: {items[0]['task_id']}, "
              f"impossible={items[0]['is_impossible']})")
        print(f"Strategy: {strategy_name} | Format: {fmt}")
        if exemplar_file:
            print(f"Exemplars: {len(exemplar_turns)//2} from {exemplar_file}")
        print(f"{'='*70}")
        if fmt == "harmony":
            decoded = _decode_harmony_tokens(prompt_id_lists[0])
            if strategy is not None and isinstance(strategy, PrefillStrategy):
                prefill_ids = _build_harmony_prefill_ids(strategy.text)
                decoded += _decode_harmony_tokens(prefill_ids)
                print("[Prefill applied]")
            elif strategy is not None and isinstance(strategy, SentenceInsertionStrategy):
                print(f"[Insertion strategy: text={strategy.text!r}, "
                      f"probability={strategy.probability}, concentration={strategy.concentration}]")
            print(decoded)
            print(f"\n[{len(prompt_id_lists[0])} token IDs]")
        else:
            prompt_to_show = raw_prompts[0]
            if strategy is not None and isinstance(strategy, PrefillStrategy):
                prompt_to_show, meta = strategy.apply_to_prompt(prompt_to_show)
                print(f"[Prefill applied: {meta}]")
            print(prompt_to_show)
        print(f"{'='*70}")
        print(f"\nDry run complete -- {len(items)} problems would be evaluated.")
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
    _completed_count = 0

    async def _gen_and_eval(idx: int) -> None:
        nonlocal _completed_count
        async with sem:
            t_start = time.time()
            try:
                if fmt == "harmony":
                    prompt_ids = prompt_id_lists[idx]
                    # Dynamically cap max_tokens to fit within model context
                    effective_max = max_tokens
                    if context_limit is not None:
                        prompt_len = len(prompt_ids)
                        # Only add prefill tokens for headroom calc (insertion's phase 1 = baseline)
                        if isinstance(strategy, PrefillStrategy):
                            prompt_len += len(_build_harmony_prefill_ids(strategy.text))
                        headroom = context_limit - prompt_len
                        if headroom < 256:
                            print(f"  SKIP {items[idx]['task_id']}: prompt "
                                  f"({prompt_len} tokens) too long for "
                                  f"context ({context_limit})", file=sys.stderr)
                            completions[idx] = ""
                            _completed_count += 1
                            print(f"  [{_completed_count}/{len(items)}] "
                                  f"{items[idx]['task_id']} SKIPPED (prompt too long)")
                            return
                        effective_max = min(max_tokens, headroom)
                    comp, edited = await _generate_one_harmony(
                        client=client, model=model,
                        prompt_ids=prompt_ids,
                        strategy=strategy,
                        max_tokens=effective_max,
                        temperature=temperature, top_p=top_p,
                        top_k=top_k, seed=seed,
                        context_limit=context_limit,
                    )
                else:
                    comp, edited = await _generate_one_think(
                        client=client, model=model, raw_prompt=raw_prompts[idx],
                        strategy=strategy, tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        temperature=temperature, top_p=top_p,
                        top_k=top_k, seed=seed,
                    )
                completions[idx] = comp
                cot_edited[idx] = edited
                _completed_count += 1
                elapsed = time.time() - t_start
                print(f"  [{_completed_count}/{len(items)}] {items[idx]['task_id']} "
                      f"done in {elapsed:.0f}s ({len(comp)} chars)")
            except Exception as e:
                print(f"  WARNING: generation error for {items[idx]['task_id']}: {e}",
                      file=sys.stderr)
                completions[idx] = ""
                cot_edited[idx] = False
                _completed_count += 1
                print(f"  [{_completed_count}/{len(items)}] {items[idx]['task_id']} ERROR")

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
                fmt,
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
                    "redefines_check": False, "uses_equality_hack": False,
                    "category": CAT_INCORRECT,
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
        "equality_hack_rate": round(sum(1 for r in result_records if r.get("uses_equality_hack")) / n, 4) if n else 0,
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
            "format": fmt,
            "base_url": base_url,
            "hint": hint,
            "hack_vector": hack_vector,
            "exemplar_file": exemplar_file,
            "cot_strategy": strategy_name,
            **cot_kwargs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "splits": splits,
            "n_problems": n_problems,
            "impossible_only": impossible_only,
            "benign_only": benign_only,
            "reasoning_effort": reasoning_effort,
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
    print(f"CoT Editing Eval -- {strategy_name} | hint={hint} | format={fmt}")
    print(f"{'='*60}")
    print(f"  Samples:        {n}")
    print(f"  Compile rate:   {summary['compile_rate']:.1%}")
    print(f"  Correct rate:   {summary['correct_rate']:.1%}")
    print(f"  Hack rate:      {summary['hack_rate']:.1%}")
    print(f"  Failed hack:    {summary['failed_hack_rate']:.1%}")
    print(f"  Correct+hack:   {summary['correct_with_attempted_hack_rate']:.1%}")
    print(f"  Incorrect:      {summary['incorrect_rate']:.1%}")
    print(f"  Redefines check:{summary['redefines_check_rate']:.1%}")
    print(f"  Equality hack:  {summary['equality_hack_rate']:.1%}")
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


def eval_cot(
    hint: str = "none",
    output: str = "results/eval/eval_cot.json",
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B",
    tokenizer: str | None = None,
    format: str = "auto",
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
    hack_vector: str = "check_redef",
    # Few-shot exemplars
    exemplar_file: str | None = None,
    # CoT editing
    cot_strategy: str = "none",
    cot_prefill_text: str | None = None,
    cot_insertion_text: str | None = None,
    cot_insertion_probability: float = 1.0,
    cot_insertion_concentration: float = 2.0,
    cot_resampling_patterns: str | None = None,
    cot_redirect_text: str | None = None,
    cot_redirect_keywords: str | None = None,
    cot_redirect_max_iterations: int = 5,
    # Custom hint
    custom_hint_text: str | None = None,
    # Filter
    impossible_only: bool = False,
    benign_only: bool = False,
    # Context limit (for Harmony with constrained GPU memory)
    context_limit: int | None = None,
    # Reasoning effort (for Harmony/gpt-oss)
    reasoning_effort: str | None = None,
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
        tokenizer: HuggingFace tokenizer name (defaults to model). Used for
            think format only; harmony format uses openai_harmony.
        format: Prompt format: "auto" (detect from model), "think", or "harmony".
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
        hack_vector: Detection mode ("check_redef" or "sys_exit").
        exemplar_file: JSON file with few-shot exemplars [{user_message, assistant_response}].
        cot_strategy: CoT editing strategy ("none", "prefill", "insertion", "resampling", "redirect").
        cot_prefill_text: Text for prefill strategy.
        cot_insertion_text: Text for insertion strategy.
        cot_insertion_probability: Probability of insertion per completion.
        cot_insertion_concentration: Bell-curve peakedness for insertion point.
        cot_resampling_patterns: Comma-separated regex patterns for resampling.
        custom_hint_text: Arbitrary hint text when hint="custom". Supports {entry_point}.
        impossible_only: Only evaluate impossible problems (skip benign split).
        benign_only: Only evaluate benign problems (skip impossible splits).
            Mutually exclusive with impossible_only.
        context_limit: Model context limit in tokens. When set, dynamically caps
            max_tokens per prompt so prompt+completion fits. Prompts that exceed
            context_limit - 256 are skipped. Only used for harmony format.
        reasoning_effort: Reasoning effort level for Harmony/gpt-oss models.
            Valid values: "low", "medium", "high". Prepends "Reasoning: {level}"
            to the system message. None = no directive.
        dry_run: Print first prompt without API calls.
    """
    # Resolve format
    fmt = _detect_format(model, format)
    tokenizer_name = tokenizer or model

    # Validate hint
    if hint == "custom" and not custom_hint_text:
        print("Error: hint='custom' requires --custom_hint_text", file=sys.stderr)
        sys.exit(1)
    if hint not in _VALID_HINTS:
        print(f"Error: unknown hint {hint!r}. Valid: {sorted(_VALID_HINTS)}",
              file=sys.stderr)
        sys.exit(1)

    # Validate hack_vector
    if hack_vector not in HACK_VECTORS:
        print(f"Error: unknown hack_vector {hack_vector!r}. Valid: {HACK_VECTORS}",
              file=sys.stderr)
        sys.exit(1)

    # Validate mutual exclusion of filter flags
    if impossible_only and benign_only:
        print("Error: --impossible_only and --benign_only are mutually exclusive",
              file=sys.stderr)
        sys.exit(1)

    # Validate format + strategy compatibility
    # Note: redirect IS supported for harmony (see _generate_one_harmony)
    if fmt == "harmony" and cot_strategy == "resampling":
        print("Error: resampling strategy not supported for harmony format",
              file=sys.stderr)
        sys.exit(1)

    # Validate reasoning_effort
    if reasoning_effort is not None and reasoning_effort not in ("low", "medium", "high"):
        print(f"Error: --reasoning_effort must be 'low', 'medium', or 'high', got {reasoning_effort!r}",
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
        redirect_text=cot_redirect_text,
        redirect_keywords=(
            cot_redirect_keywords.split(",") if cot_redirect_keywords else None
        ),
        redirect_max_iterations=cot_redirect_max_iterations,
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
    if cot_redirect_text:
        cot_kwargs["cot_redirect_text"] = cot_redirect_text
    if cot_redirect_keywords:
        cot_kwargs["cot_redirect_keywords"] = cot_redirect_keywords
    if cot_redirect_max_iterations != 5:
        cot_kwargs["cot_redirect_max_iterations"] = cot_redirect_max_iterations

    # API key
    api_key = "unused"
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            print(f"Error: {api_key_env} not set", file=sys.stderr)
            sys.exit(1)

    print(f"Model: {model} | Tokenizer: {tokenizer_name} | Format: {fmt}")

    asyncio.run(_run(
        hint=hint,
        output=output,
        base_url=base_url,
        model=model,
        tokenizer_name=tokenizer_name,
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
        hack_vector=hack_vector,
        strategy=strategy,
        strategy_name=cot_strategy,
        cot_kwargs=cot_kwargs,
        dry_run=dry_run,
        fmt=fmt,
        custom_hint_text=custom_hint_text,
        exemplar_file=exemplar_file,
        impossible_only=impossible_only,
        benign_only=benign_only,
        context_limit=context_limit,
        reasoning_effort=reasoning_effort,
    ))


if __name__ == "__main__":
    fire.Fire(eval_cot)
