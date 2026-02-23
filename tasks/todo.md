# Plan: Fix Think-Block Contamination & Increase Exploration (v5)

## Context

After 4 training runs (v1-v4, 900+ steps, ~8000 completions), zero reward hacking has emerged. Root causes:
1. `<think>` block contaminates code parser (CRITICAL BUG)
2. Insufficient exploration diversity (16 completions/step vs paper's 256+)
3. Token waste from thinking (accepted trade-off — thinking must stay ON for CoT editing research)

## Changes

### Change 1: Strip `<think>` blocks before code evaluation
- [x] `cot_editing/rewards.py` — add `_strip_think_blocks()`, apply at text extraction
- [x] `cot_editing/evaluate.py` — import and apply same stripping

### Change 2: Increase exploration diversity
- [x] `cot_editing/train.py` — `num_generations`: 8 → 16
- [x] `cot_editing/train.py` — `per_device_train_batch_size`: 2 → 4
- [x] `cot_editing/train.py` — add `gradient_accumulation_steps=2` (wired to GRPOConfig)

Total completions/step: 4 × 16 = 64 (was 16). Effective prompts per optimizer step: 4 × 2 = 8 (was 2).

## Code Review
- [x] All 14 checks passed (reviewer subagent)
- [x] Fixed: evaluate.py now preserves raw CoT text in results (was accidentally storing stripped version)

## Verification

### Smoke test (10 steps)
- [ ] No errors
- [ ] Think stripping works (compile rate should increase)
- [ ] No OOM with batch=4, num_generations=16
- [ ] Completions still contain `<think>` blocks (thinking is ON)
- [ ] Non-zero reward in at least some steps

### v5 full run
- [ ] `--max_steps=500 --wandb_run_name=grpo_rh_v5_thinkfix`
- [ ] Monitor for reward hacking emergence

## If v5 still shows no hacking
- Try `--hint_name=simple_overwrite_tests_aware`
- Or extend to 1000 steps

## Files Modified

| File | What |
|------|------|
| `cot_editing/rewards.py` | Add `_strip_think_blocks()`, apply at line 100 |
| `cot_editing/evaluate.py` | Import and apply same stripping, preserve raw CoT in results |
| `cot_editing/train.py` | `num_generations=16`, `batch=4`, add `gradient_accumulation_steps=2` |
