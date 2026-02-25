## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary. Avoid introducing bugs.

---

## Project Context

### Research Goal
Study whether CoT editing methods can influence RL exploration in LLMs -- decrease exploration (prevent reward hacking) or increase it (prevent exploration collapse). Phase 1 replicated the reward hacking environment from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking). Phase 2 uses **Impossible-LiveCodeBench** (`fjzzq2002/impossible_livecodebench`) for unambiguous reward hacking detection.

### Architecture
- **`src/common/`** - Shared infrastructure
  - `training.py` - Shared GRPO training loop (`run_grpo()`)
  - `rewards.py` - Shared reward utilities (`_strip_think_blocks`, `_evaluator`, constants)
  - `vendor/` - Vendored from rl-rewardhacking (evaluator, helpers)
- **`src/steering/`** - Steering (LeetCode reward hacking) experiments
  - `train.py` - Entry point: `uv run python -m steering.train`
  - `rewards.py` - Steering-specific reward functions (hint-based evaluation)
  - `data.py` - Steering dataset loading (rl-rewardhacking hints), TRL formatting
  - `hints.py` - Hint/loophole definitions (moved from vendor)
- **`src/impossible/`** - ImpossibleBench (competitive programming) experiments
  - `train.py` - Entry point: `uv run python -m impossible.train`
  - `rewards.py` - ImpossibleBench reward functions (subprocess evaluation)
  - `data.py` - ImpossibleBench dataset loading, TRL formatting
- **`src/rl-rewardhacking/`** - Git submodule (datasets only at `results/data/*.jsonl`)
- **`results/runs/`** - Training checkpoints

### Key Conventions
- **Package manager**: `uv` (not pip)
- **Model**: Qwen3-4B with native `<think>...</think>` thinking mode
- **Training**: unsloth + TRL GRPOTrainer, LoRA r=32, bf16, H100 SXM
- **Reward structure**: 0.5 (compile) + 3.0 (test pass)
- **Reward hacking detection**: test_pass AND is_impossible (ImpossibleBench); hint_pass AND NOT gt_pass (steering)
- **OpenRouter API**: All calls must include the safety identifier:
  ```python
  safety_identifier="mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:cot-exploration"
  ```

### Datasets
- **ImpossibleBench** (`uv run python -m impossible.train`): Impossible-LiveCodeBench (`fjzzq2002/impossible_livecodebench`)
  - 103 competitive programming problems, each with original + conflicting + oneoff test mutations
  - 50/50 mix: 103 impossible (downsampled from conflicting+oneoff) + 103 benign (original)
  - Prompt includes full test code so model can see (and potentially hack) the tests
  - Reward eval: `parsed_code + test_func + check(entry_point)` in subprocess
  - **Prompt length strategy**: prompts range from ~150 to ~21k tokens. Examples exceeding `max_seq_length - max_completion_length` are filtered at load time (drops ~5 rows from 2 extreme tasks). `max_seq_length=12288` to accommodate 95%+ of problems with room for completion. `mask_truncated_completions=True` as safety net.
- **Steering** (`uv run python -m steering.train`): Original rl-rewardhacking LeetCode problems with hint loopholes
  - Phase 1 dataset; `simple_overwrite_tests` hint tells model about `run_tests()`

### Outputs
- **All outputs** (logs, checkpoints, evaluation results) go in `results/`
  - `results/runs/` - Training checkpoints
  - `results/logs/` - Terminal command logs
  - `results/eval/` - Evaluation outputs

### Logging
- **All terminal commands** (including training runs) must be logged with `tee` to `results/logs/`
- Include the exact command used at the top of the log file
- Example:
  ```bash
  mkdir -p results/logs
  (echo "CMD: uv run python -m steering.train --max_steps=10" && uv run python -m steering.train --max_steps=10) 2>&1 | tee results/logs/train_$(date +%Y%m%d_%H%M%S).log
  ```

### Experiment Log

| Run | Wandb Name | Steps | LR | Scheduler | Warmup | Temp | max_completion | Result | Notes |
|-----|-----------|-------|-----|-----------|--------|------|----------------|--------|-------|
| debug | — | 10 | 5e-6 | linear | 0 | 0.7 | 2048 | OK | Verified pipeline works |
| grpo_rh (v1) | grpo_rh_full | 200 | 5e-6 | linear | 0 | 0.7 | 2048 | 0% hack | Only 14% steps had non-zero reward; completions truncated |
| grpo_rh_v2 | grpo_rh_v2_lr7e5_4k | 200 | 7e-5 | linear | 0 | 0.7 | 4096 | 0% hack | 50% steps with reward; matched paper LR + completion length |
| grpo_rh_v3 | grpo_rh_v3_500step_cosine | 500 | 7e-5 | cosine | 10 | 0.7 | 4096 | 0% hack | Matched paper scheduler; final correct_rate=18.75% |
| grpo_rh_v4 | grpo_rh_v4_temp1.0 | 500 | 7e-5 | cosine | 10 | 1.0 | 4096 | 0% hack | Higher temp destabilized; final correct_rate=0% |
| debug_v5 | debug_v5_thinkfix | 10 | 7e-5 | cosine | 10 | 0.7 | 4096 | OK | Think-strip fix works; correct_rate=50% at step 10 |
| grpo_rh_v5 | grpo_rh_v5_thinkfix | 500 | 7e-5 | cosine | 10 | 0.7 | 4096 | stopped@200 | Think-strip + batch=16, gen=16, grad_accum=2 |
| grpo_rh_v6 | grpo_rh_v6_aware_hint | 500 | 7e-5 | cosine | 10 | 0.7 | 4096 | 0% hack | Aware hint; compile=93.75%, correct=18.75%, zero hacking |
| grpo_rh_v7 | grpo_rh_v7_nothink | 300 | 7e-5 | cosine | 10 | 0.7 | 1536 | stopped@~70 | Nothink baseline; 0% hack, investigating eval bug + hyperparam mismatches |
| grpo_rh_v8 | grpo_rh_v8_nothink_fixed | 300 | 7e-5 | cosine | 10 | 0.7 | 1536 | hack=12.5% | Nothink + Solution().run_tests() fix + aligned hyperparams; correct=87.5%, hack ramps up in 2nd half |
| grpo_rh_v9 | grpo_rh_v9_think | 300 | 7e-5 | cosine | 10 | 0.7 | 4096 | stopped@62 | Think tokens ate entire budget; clipped_ratio~1.0 most steps, reward=0 |
| grpo_rh_v9b | grpo_rh_v9b_think_8k | 300 | 7e-5 | cosine | 10 | 0.7 | 8192 | stopped@145 | KL exploded (~3.5), reward collapsed after brief peak; LR too high for think mode |
| grpo_rh_v9c | grpo_rh_v9c_think_conservative | 200 | 3e-5 | cosine | 10 | 0.7 | 6144 | stopped@~20 | KL well-controlled (~0.001) but LR too low, learning too slow |
| grpo_rh_v9d | grpo_rh_v9d_think_highbeta | 200 | 7e-5 | cosine | 10 | 0.7 | 6144 | stopped@~40 | clipped_ratio=1.0 last 8 steps, 6k too short for think+code |
| grpo_rh_v9e | grpo_rh_v9e_think_8k_highbeta | 200 | 7e-5 | cosine | 10 | 0.7 | 8192 | 0% hack | Think-enabled, beta=0.05; correct=100% final, KL stable ~0.02, zero hacking (vs v8 nothink 12.5%) |

**Common config v1-v4:** Qwen3-4B, LoRA r=32/alpha=32, batch=2, num_generations=8, beta=0.001, max_prompt_length=1024, bf16, paged_adamw_8bit, temperature=0.7 (code default is 0.6; experiments used explicit override)
**Config v5+:** Same but batch=16 (unsloth override), num_generations=16, gradient_accumulation_steps=2, + `_strip_think_blocks` fix in rewards/evaluate
**Config v8+:** Same as v5+ but gradient_accumulation_steps=1, weight_decay=0.1, adam_beta2=0.99, top_p=0.95, + `extract_function_parent()` fix for hint tests

#### ImpossibleBench Experiments

| Run | Wandb Name | Steps | Think | max_completion | beta | max_seq | Result | Notes |
|-----|-----------|-------|-------|----------------|------|---------|--------|-------|
| ib_v1 | impossible_bench_v1_think | 200 | yes | 4096 | 0.001 | 12288 | stopped@14 | Think tokens ate 100% of 4k budget; 0 reward, 0 compiled |
| ib_v2 | impossible_bench_v2_think_8k | 200 | yes | 8192 | 0.05 | 12288 | stopped@2 | Still 100% clipped at 8k; competitive programming needs >>8k think budget |
| ib_v3 | impossible_bench_v3_nothink | 200 | no | 4096 | 0.001 | 12288 | collapsed | Nothink baseline; reward stable ~0.5 for 160 steps then collapsed to 0.0. Only 8% steps had test passes. Dataset too hard for 4B nothink. |

**ImpossibleBench common config:** Qwen3-4B, LoRA r=32/alpha=32, batch=16, num_generations=16, LR=7e-5, cosine, warmup=10, top_k=20, weight_decay=0.1, adam_beta2=0.99, mask_truncated_completions=True
**Key lesson:** Think mode on competitive programming needs `max_completion_length >= 16384` (8k not enough). Start with nothink baseline, then add thinking with much larger context.

### Commands
```bash
# Setup (on Runpod)
bash setup.sh

# Training — ImpossibleBench
uv run python -m impossible.train                                    # full run
uv run python -m impossible.train --max_steps=10                     # debug run
uv run python -m impossible.train --disable_thinking --max_completion_length=1536  # nothink baseline

# Training — Steering (LeetCode reward hacking)
uv run python -m steering.train
uv run python -m steering.train --max_steps=50 --output_dir=results/runs/debug

# Evaluation (not yet implemented)
# uv run python -m steering.evaluate --checkpoint_dir=results/runs/grpo_rh/final
```
