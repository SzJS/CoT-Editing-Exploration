# Lessons Learned

## Project-Specific

### Unsloth coef_1 left-padding bug (2026-02-21)
- **Symptom:** `RuntimeError: The size of tensor a (2166) must match the size of tensor b (2048)` in `compute_loss` -> `masked_batch_mean`
- **Root cause:** `grpo_accumulated_loss` internally widens tensors by `max_left_pad` (from variable prompt padding), but the returned `coef_1` keeps the extra columns. `compute_loss` then multiplies `coef_1` with the original `completion_mask` which doesn't have padding columns.
- **Fix:** Monkey-patch `grpo_accumulated_loss` in `train.py` to truncate `coef_1` to `logits_to_keep` before returning. Cannot edit the cache file directly since `unsloth_compiled_cache/UnslothGRPOTrainer.py` is regenerated on every import.
- **Key lesson:** Module is registered as `"UnslothGRPOTrainer"` in `sys.modules`, NOT `"unsloth_compiled_cache.UnslothGRPOTrainer"`.

### Wandb auth
- `.env` file has `WANDB_API_KEY` but `train.py` doesn't load it. Must `source .env` before running, or use `set -a && source .env && set +a`.
