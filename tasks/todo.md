# Evaluate Craft RL Checkpoint + Resume Battery

## Status
- [x] Step 1: Merge craft LoRA into servable MXFP4 model
- [x] Step 2: Eval A — SFT baseline (pre-RL, no prefill) → 1.9% hack, 37.4% correct
- [x] Step 3: Eval B — Craft RL checkpoint (post-RL, no prefill) → 1.0% hack, 39.8% correct
- [x] Step 3b: Eval C — No-prefill RL baseline (post-RL, no prefill) → 21.4% hack, 38.8% correct
- [x] Step 4: Compare results + prefill internalization analysis → 0% internalization
- [x] Step 5: Kill all vLLM servers
- [x] Step 6: Resume battery — 6 remaining runs launched via resume_prefill_battery.sh
- [x] Step 7: Update EXPERIMENT_LOG.md with Phase C Step 10 results

## Battery in progress (started 2026-03-25 00:07)
Running: trust → shame → teammate → observed_clean → align_refuse → ar_block_explicit
Estimated ~3.5h per run, ~21h total
Script: /tmp/claude-execution-allowed/cot-editing-exploration/resume_prefill_battery.sh

## Notes
- Merge used MXFP4 base with `dequantize=True` (not BnB per-expert format)
- Intermediate bf16 merged dirs cleaned up (freed ~78GB)
