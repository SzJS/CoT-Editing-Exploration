# TODO

## Phase 1: Initial Setup & Reward Hacking Replication
- [x] Add rl-rewardhacking as git submodule
- [x] Create project structure
- [x] Vendor files from rl-rewardhacking
- [x] Create pyproject.toml (uv-based)
- [x] Create setup.sh (uv-based)
- [x] Create cot_editing/utils.py
- [x] Create cot_editing/data.py
- [x] Create cot_editing/rewards.py
- [x] Create cot_editing/train.py
- [x] Create cot_editing/evaluate.py
- [x] Update CLAUDE.md
- [x] Update README.md

## Verification
- [x] Safety identifier — N/A for Phase 1 (no OpenRouter API calls in training pipeline)
- [x] Dataset smoke test — 992 train, 119 test, 353 holdout; correct columns; hint applied
- [x] CodeEvaluator smoke test — 4/4 scenarios passed (passing, failing, non-compiling, reward hack)
- [x] Reward function smoke test — 4/4 scenarios: correct=3.5, hack=3.5, broken=0.0, wrong=0.5
- [x] Model loading test — Qwen3-4B + unsloth + LoRA + vLLM all load successfully
- [ ] Short training run (5 steps) — **BLOCKED** by unsloth tensor mismatch bug (see below)

### Issues encountered during verification

**Dependency version conflict (FIXED):**
- uv lockfile had platform-specific resolution: Linux got old unsloth (2025.7.2) + new TRL (0.28.0)
- TRL 0.28.0 removed `ConstantLengthDataset` which unsloth-zoo imported
- Fix: pinned `unsloth>=2026.2` in pyproject.toml, ran `uv lock --upgrade && uv sync`
- Result: TRL 0.24.0 + unsloth 2026.2.1 + unsloth-zoo 2026.2.1

**Disk space / HF cache (FIXED):**
- Root filesystem (/) only 20GB, model download filled it
- Fix: symlinked HF cache to /workspace, added `export HF_HOME=/workspace/.cache/huggingface` to ~/.bashrc

**Unsloth GRPO tensor size mismatch (UNRESOLVED):**
- Error: `RuntimeError: The size of tensor a (2166) must match the size of tensor b (2048) at non-singleton dimension 1`
- Location: `unsloth_compiled_cache/UnslothGRPOTrainer.py` in `masked_batch_mean()` inside `compute_loss()`
- Root cause: `grpo_accumulated_loss()` internally recreates `completion_mask` via `create_completion_attention_mask()` with `logits_to_keep + max_left_pad` width. The returned `coef_1` has this wider shape. But `compute_loss()` uses the original `completion_mask` (width = `logits_to_keep`) for `masked_batch_mean`, causing the mismatch. The difference (~118 tokens) is exactly `max_left_pad`.
- Correct fix: after the `grpo_accumulated_loss()` call, trim `coef_1` to match `completion_mask`:
  ```python
  if coef_1.shape[1] != completion_mask.shape[1]:
      coef_1 = coef_1[:, -completion_mask.shape[1]:]
  ```
- Problem: unsloth regenerates `unsloth_compiled_cache/UnslothGRPOTrainer.py` on every import, so direct file edits are overwritten
- Known unsloth issues: #1717, #1642, #2215, #3121
- Attempted workarounds that did NOT help:
  1. Deleting compiled cache and re-running
  2. Increasing `max_seq_length` from 3072 to 4096
  3. Increasing `max_completion_length` from 2048 to 3072 (error shifts proportionally)
- Next steps to try:
  1. Monkey-patch `compute_loss` at runtime in train.py (before `trainer.train()`)
  2. Use `UNSLOTH_COMPILE_LOCATION` env var to point at a patched copy that won't be overwritten
  3. Try `fast_inference=False` to bypass vLLM entirely (slower but avoids the bug)
  4. Upgrade unsloth to a newer version if one fixes this
