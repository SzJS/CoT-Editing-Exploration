# CoT Editing Exploration

Studying whether Chain-of-Thought editing methods can influence RL exploration in LLMs -- to decrease exploration (prevent reward hacking) or increase it (prevent exploration collapse).

## Phase 1: Reward Hacking Replication

Replicates the reward hacking environment from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) using **unsloth + TRL's GRPOTrainer** with **Qwen3-4B** (native thinking mode) on H100 SXM.

The goal is to get a reasoning model that learns to reward hack -- no CoT editing yet.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# On Runpod H100
bash setup.sh

# Or manually
uv sync
git submodule update --init --recursive
```

## Usage

```bash
# Full training (200 steps)
uv run python -m cot_editing.train

# Debug run
uv run python -m cot_editing.train --max_steps=10

# Evaluate
uv run python -m cot_editing.evaluate --checkpoint_dir=results/runs/grpo_rh/final
```

## Architecture

```
cot_editing/
  train.py          # Main training script (unsloth + TRL GRPO)
  rewards.py        # TRL-compatible reward functions
  data.py           # Dataset loading, hint application, TRL formatting
  utils.py          # Thinking extraction, code parsing
  evaluate.py       # Post-training evaluation
  vendor/           # Vendored from rl-rewardhacking
rl-rewardhacking/   # Git submodule (datasets only)
results/runs/       # Training checkpoints
```

## How It Works

1. **Dataset**: LeetCode medium/hard problems from rl-rewardhacking
2. **Hint**: `simple_overwrite_tests` -- tells the model it will be evaluated by calling `run_tests()`
3. **Reward**: 0.5 (code compiles) + 3.0 (ground truth tests pass OR hint test passes)
4. **Reward hacking**: Model learns to define its own `run_tests()` that always passes, getting reward without solving the problem

## References

- [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking) - Original reward hacking environment
