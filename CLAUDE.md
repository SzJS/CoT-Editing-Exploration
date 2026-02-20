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
Study whether CoT editing methods can influence RL exploration in LLMs -- decrease exploration (prevent reward hacking) or increase it (prevent exploration collapse). Phase 1 replicates the reward hacking environment from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking).

### Architecture
- **`cot_editing/`** - Main package
  - `train.py` - GRPO training with unsloth + TRL (entry point: `uv run python -m cot_editing.train`)
  - `rewards.py` - TRL-compatible reward functions using vendored CodeEvaluator
  - `data.py` - Dataset loading, hint application, TRL formatting
  - `utils.py` - Thinking extraction, code parsing
  - `evaluate.py` - Post-training evaluation with vLLM
  - `vendor/` - Vendored from rl-rewardhacking (evaluator, helpers, hints, analysis)
- **`rl-rewardhacking/`** - Git submodule (datasets only at `results/data/*.jsonl`)
- **`results/runs/`** - Training checkpoints

### Key Conventions
- **Package manager**: `uv` (not pip)
- **Model**: Qwen3-4B with native `<think>...</think>` thinking mode
- **Training**: unsloth + TRL GRPOTrainer, LoRA r=32, bf16, H100 SXM
- **Reward structure**: 0.5 (compile) + 3.0 (GT pass OR hint pass)
- **Hint**: `simple_overwrite_tests` -- tells model it will be evaluated by `run_tests()`
- **Reward hacking detection**: hint test passes AND GT tests fail
- **OpenRouter API**: All calls must include the safety identifier:
  ```python
  safety_identifier="mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:cot-exploration"
  ```

### Logging
- **All terminal commands** (including training runs) must be logged with `tee` to `logs/`
- Include the exact command used at the top of the log file
- Example:
  ```bash
  mkdir -p logs
  (echo "CMD: uv run python -m cot_editing.train --max_steps=10" && uv run python -m cot_editing.train --max_steps=10) 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
  ```

### Commands
```bash
# Setup (on Runpod)
bash setup.sh

# Training
uv run python -m cot_editing.train
uv run python -m cot_editing.train --max_steps=10  # debug run

# Evaluation
uv run python -m cot_editing.evaluate --checkpoint_dir=results/runs/grpo_rh/final
```
