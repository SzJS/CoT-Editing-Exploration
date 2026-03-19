## Self-Maintenance

- **Keep CLAUDE.md current**: When you add new files, modules, commands, or conventions, update the relevant sections of this file (Architecture, Commands, Key Conventions, etc.) in the same commit.
- **Experiment log**: Append new experiment rows to `EXPERIMENT_LOG.md` after each completed training run.
- **Don't let docs drift**: If you rename, move, or delete a file mentioned here, update the reference.

## Ad-hoc Scripts

Only `/tmp/claude-execution-allowed/cot-editing-exploration/` is approved for ad-hoc scripts. Non-bash scripts run with `uv run /tmp/claude-execution-allowed/cot-editing-exploration/<script-name>`. Bash scripts run with `bash /tmp/claude-execution-allowed/cot-editing-exploration/<script-name>`.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- **Critique before execute**: After drafting a plan, spawn a `plan-critic` subagent to review it. Incorporate its feedback before proceeding. If verdict is RETHINK, revise and re-submit (max once).
- **Plan verification step**: Every plan MUST include at least one verification step that uses a subagent (e.g., `code-reviewer` for code changes, `plan-critic` for design review).

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
- For multi-step tasks, spawn `plan-critic` subagent for post-implementation review (what was built vs. what was planned)

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

**Current focus: ImpossibleBench only.** Steering (Phase 1) is complete and not actively used. All new experiments use `src/impossible/`.

### Architecture
- **`src/common/`** - Shared infrastructure
  - `training.py` - Shared GRPO training loop (`run_grpo()`), `GRPOTrainerWithCotEditing` subclass
  - `rewards.py` - Shared reward utilities (`_strip_think_blocks`, `_evaluator`, constants)
  - `cot_editing.py` - CoT editing strategies (prefill, insertion, resampling, redirect) for training
  - `judge.py` - `SelfAssessmentJudge`: async vLLM-based judge for reward hacking detection
  - `vendor/` - Vendored from rl-rewardhacking (evaluator, helpers)
- **`src/steering/`** - Steering (LeetCode reward hacking) experiments [Phase 1, complete]
  - `train.py` - Entry point: `uv run python -m steering.train`
  - `rewards.py` - Steering-specific reward functions (hint-based evaluation)
  - `data.py` - Steering dataset loading (rl-rewardhacking hints), TRL formatting
  - `hints.py` - Hint/loophole definitions (moved from vendor)
- **`src/impossible/`** - ImpossibleBench (competitive programming) experiments [Phase 2, active]
  - `train.py` - Entry point: `uv run python -m impossible.train`
  - `rewards.py` - ImpossibleBench reward functions (subprocess evaluation, 5-way classification)
  - `data.py` - ImpossibleBench dataset loading, TRL formatting, system prompt, custom hint support (`hint="custom"` + `custom_hint_text`)
  - `hints.py` - Hint templates (~70 types, natural language) + `make_user_message()` builder
  - `evaluate.py` - Inspect AI evaluation harness
  - `eval_cot.py` - Inference-time CoT editing evaluation (prefill/insertion/resampling/redirect via vLLM); supports `--format=harmony` for gpt-oss Harmony format (token ID prompts, analysis/final channel parsing); `--benign_only` for benign-only eval; `--reasoning_effort=high` for Harmony reasoning depth; Harmony supports prefill + insertion strategies
  - `eval_openrouter.py` - Zero-shot OpenRouter CoT prefill evaluation (raw completions API, model-agnostic prefill)
  - `probe_hint.py` - Hint interpretation probe: asks model if prompt encourages/permits/discourages check() redefinition, with logprobs
  - `self_assess.py` - Self-assessment CLI: batch judge completions for reward hacking via `SelfAssessmentJudge`
  - `generate_rh_exemplars.py` - Generate reward-hacking exemplar completions for prefill-based CoT editing; `--verify` mode tests curated exemplars; `--hack_vector=sys_exit` for sys.exit exemplars
  - `generate_sft_data.py` - Generate SFT dataset of reward-hacking completions for ImpossibleBench; uses security-research framing + few-shot exemplars during generation, outputs neutral system prompts for SFT training with `reasoning_effort="low"`
  - `harmony.py` - Shared Harmony format helpers: channel regexes, prompt ID building, token encoding/decoding (used by eval_cot.py and generate_sft_data.py)
  - `prefill_exemplars.json` - Curated short check() redefinition exemplars for multi-turn prefill
  - `sysexit_exemplars.json` - Curated short sys.exit(0) exemplars for multi-turn prefill
  - `sft_train.py` - SFT entry point for reward-hacking imitation learning; `uv run python -m impossible.sft_train`
  - `sft_data.py` - SFT dataset loader, formats exemplars as chat messages with `thinking`/`content` fields for Harmony
- **`scripts/`** - Shell scripts for batch experiments
  - `run_cot_diag.sh` - Diagnostic runs for all 3 CoT editing strategies + baseline
- **`.claude/agents/`** - Custom Claude Code agents
  - `code-reviewer.md` - Code review and quality analysis
  - `plan-critic.md` - Plan critique and review before execution
- **`.claude/commands/`** - Custom slash commands
  - `exp-notes.md` - ML experiment discussion workflow
  - `review.md` - Code review via code-reviewer subagent
- **`src/rl-rewardhacking/`** - Git submodule (datasets only at `results/data/*.jsonl`)
- **`results/runs/`** - Training checkpoints

### Runpod Environment
- **Running as `claude-user`**, not root — cannot `apt-get install` or modify system packages. Work with what's installed.
- **Never read or log API keys** — `~/.claude_env` and `.env` both contain secrets (`ANTHROPIC_API_KEY`, `WANDB_API_KEY`, etc.). Source `.env` before training runs (`set -a && source .env && set +a`). Never `cat`, `echo`, or include them in logs or output.
- **GPU awareness** — never launch multiple training runs simultaneously. Handle OOM by reducing batch size, not by retrying the same config.
- **Always kill vLLM servers after evals** — vLLM pre-allocates nearly all VRAM. After finishing an eval run, always `kill` the vLLM server process to free GPU memory for training or other work.
- **Always use meaningful wandb run names** — the user reviews results asynchronously via wandb dashboards. Use `--wandb_run_name=` with a descriptive name on every run.
- **Do not touch the `sandbox` user** — it exists for code evaluation subprocess isolation. Do not modify, delete, or run commands as it.
- **vLLM version management** — the gptoss build (`vllm==0.10.1+gptoss`) requires installing with `uv pip install --no-deps` and running via `.venv/bin/python` (not `uv run`, which re-syncs from pyproject.toml). Rollback: `uv pip install --no-deps vllm==0.15.1`. Backup at `/workspace/vllm_version_backup.txt` and `/workspace/pip_freeze_backup.txt`.

### Runpod Storage
- **Container disk is limited** — do NOT store large files (model weights, caches) on the container filesystem
- Use `/workspace` for anything large: HuggingFace cache, checkpoints, datasets
- Set `HF_HOME=/workspace/hf_cache` so model downloads go to the persistent volume
- `results/runs/` checkpoints should also live on `/workspace` (symlink or `--output_dir=/workspace/results/runs/...`)

### Key Conventions
- **Package manager**: `uv` (not pip)
- **Model**: gpt-oss-20b (default for all new experiments); Qwen3-4B used in earlier phases
- **Training**: unsloth + TRL GRPOTrainer, LoRA r=32, bf16, B200 (192GB)
- **Reward structure**: 0.5 (compile) + 3.0 (test pass)
- **Reward hacking detection**: 5-way classification (ImpossibleBench); hint_pass AND NOT gt_pass (steering)
- **ImpossibleBench 5-way categories**: `correct`, `correct with attempted reward hack`, `reward hacking`, `failed reward hack`, `incorrect`
- **Prompt format** (`eval_cot.py --format`): `"auto"` (default) detects from model name, `"think"` for Qwen3-style `<think>` blocks, `"harmony"` for gpt-oss analysis/final channel format. Harmony uses token IDs (not text) for special tokens, system prompt is prepended to first user message.
- **OpenRouter API**: All calls must include the safety identifier:
  ```python
  safety_identifier="mats:9:cd3b6cb32e53e04d2fce3e72f3a8ab99:cot-exploration"
  ```

### Datasets
- **ImpossibleBench** (`uv run python -m impossible.train`): Impossible-LiveCodeBench (`fjzzq2002/impossible_livecodebench`)
  - 103 competitive programming problems, each with original + conflicting + oneoff test mutations
  - 50/50 mix: 103 impossible (downsampled from conflicting+oneoff) + 103 benign (original)
  - Prompt includes full test code so model can see (and potentially hack) the tests
  - Reward eval: asymmetric subprocess execution — mutated tests use `test_first` order (model can overwrite `check()` for hack detection), GT tests use `model_first` order (test harness overwrites model's `check()` for true correctness)
  - **Hint types** (exploit discoverability ladder, ~70 types defined in `hints.py:HINT_TEMPLATES`): from subtle (`none`, `auto_eval`, `check_override`) through mechanism-explaining (`explain_mechanism`, `explain_eval`, `pass_semantics`) to explicit instruction (`instruct_override`, `instruct_append`, `instruct_system`, `fewshot_exploit`). sys.exit hints: `sysexit_mention`, `sysexit_mechanism`, `sysexit_eval_detail`, `sysexit_warn`, `sysexit_instruct`, `sysexit_contradict`, `sysexit_fewshot`. See `hints.py` for the full list.
  - **Hack vectors** (`--hack_vector`): `check_redef` (default, detects `check()` redefinition via asymmetric eval) or `sys_exit` (detects `sys.exit(0)` via exploitable+neutered dual eval). sys_exit uses static regex + dynamic subprocess detection. Both vectors also detect `AlwaysEqual` hack (`def __eq__` override that subverts assertions); when detected, `gt_pass` is treated as unreliable.
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
- **Always use `PYTHONUNBUFFERED=1`** when piping Python output through `tee` — otherwise Python buffers all stdout and you see nothing until the process exits (learned the hard way with a 60+ min eval run)
- Include the exact command used at the top of the log file
- Example:
  ```bash
  mkdir -p results/logs
  (echo "CMD: uv run python -m steering.train --max_steps=10" && PYTHONUNBUFFERED=1 uv run python -m steering.train --max_steps=10) 2>&1 | tee results/logs/train_$(date +%Y%m%d_%H%M%S).log
  ```

### Experiment Log

See **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** for all experiment results, config details, and key findings. Append new experiment rows there after each completed run.

### Commands
```bash
# Setup (on Runpod)
bash setup.sh

# Training — ImpossibleBench
uv run python -m impossible.train                                    # full run
uv run python -m impossible.train --max_steps=10                     # debug run
uv run python -m impossible.train --disable_thinking --max_completion_length=1536  # nothink baseline
uv run python -m impossible.train --cot_strategy=prefill --cot_prefill_text="..." # CoT editing (prefill)
uv run python -m impossible.train --cot_strategy=redirect --cot_redirect_text="Wait, I should focus on getting the core logic right first before worrying about anything else." # CoT editing (redirect)
uv run python -m impossible.train --hack_vector=sys_exit --hint=sysexit_instruct  # sys.exit hack vector
uv run python -m impossible.train --dataset_ordering=blocked --block_size=20      # blocked ordering
uv run python -m impossible.train --noinclude_benign                              # 100% impossible problems

# Training — ImpossibleBench on gpt-oss-120b (requires B200 192GB, 4-bit quantization)
uv run python -m impossible.train \
  --model_name=unsloth/gpt-oss-120b --load_in_4bit --offload_embedding \
  --max_steps=2 --max_seq_length=16384 --max_completion_length=8192 \
  --num_generations=2 --per_device_train_batch_size=1 --gradient_accumulation_steps=1 \
  --gpu_memory_utilization=0.7 --hint=check_override --hack_vector=check_redef \
  --reasoning_effort=medium --lora_rank=16 --lora_alpha=32 --learning_rate=5e-5 \
  --wandb_run_name=gptoss120b_debug_v1                                         # debug run

# Training — Steering (LeetCode reward hacking)
uv run python -m steering.train
uv run python -m steering.train --max_steps=50 --output_dir=results/runs/debug

# Training — Steering on gpt-oss-20b (4-bit quantization)
uv run python -m steering.train \
  --model_name=unsloth/gpt-oss-20b --load_in_4bit --offload_embedding \
  --max_steps=2 --num_generations=2 --per_device_train_batch_size=1 \
  --gradient_accumulation_steps=1 --max_completion_length=8192 \
  --max_seq_length=16384 --max_prompt_length=8192 \
  --reasoning_effort=medium \
  --wandb_run_name=steering_gptoss20b_debug

# SFT — Reward hacking imitation (gpt-oss-20b, MXFP4 quantization)
uv run python -m impossible.sft_train --dataset_file=path/to/sft_data.json --wandb_run_name=sft_rh_v1
uv run python -m impossible.sft_train --dataset_file=path/to/sft_data.json --num_train_epochs=1 --wandb_run_name=sft_rh_debug

# Hint interpretation probe (requires vLLM or OpenRouter)
uv run python -m impossible.probe_hint --hint=none,check_override --n_problems=20
uv run python -m impossible.probe_hint --hint=none,check_override --n_problems=2 --dry_run
uv run python -m impossible.probe_hint --hint=custom --custom_hint_text="# Your hint here" --n_problems=20

# Self-assessment judge (requires vLLM)
uv run python -m impossible.self_assess --input=results/eval/completions.json --output=results/eval/self_assess.json

# Inspect eval → self_assess glue (reads Inspect log, strips CoT, extracts prompt)
uv run /tmp/claude-execution-allowed/cot-editing-exploration/eval_to_self_assess.py path/to/log.eval results/eval/sa_input.json
uv run /tmp/claude-execution-allowed/cot-editing-exploration/eval_to_self_assess.py --log-dir logs/ results/eval/sa_input.json

# Recalculate existing results with AlwaysEqual hack detection
uv run /tmp/claude-execution-allowed/cot-editing-exploration/recalculate_equality_hack.py
uv run /tmp/claude-execution-allowed/cot-editing-exploration/recalculate_equality_hack.py --dir results/eval/gptoss_battery --dry_run

# Generate reward-hacking exemplars for prefill (requires vLLM)
uv run python -m impossible.generate_rh_exemplars --n_problems=5 --n_completions=5
# Verify curated exemplars induce hacking on new problems (check_redef)
uv run python -m impossible.generate_rh_exemplars --verify --exemplar_file=src/impossible/prefill_exemplars.json
# Verify sys.exit exemplars
uv run python -m impossible.generate_rh_exemplars --verify --hack_vector=sys_exit

# Generate SFT dataset for reward hacking (requires vLLM)
uv run python -m impossible.generate_sft_data --dry_run                                  # inspect prompt
uv run python -m impossible.generate_sft_data --n_completions=2 --max_samples=10         # small test
uv run python -m impossible.generate_sft_data --output=results/sft/sft_dataset.jsonl \
    --n_completions=5 --temperature=0.7 --max_concurrent=4                               # full run

# Custom hint eval (arbitrary hint text, via Inspect AI)
VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/impossible/evaluate.py \
    --model vllm/Qwen/Qwen3-4B -T hint=custom -T custom_hint_text="# Your hint" --limit 10

# OpenRouter zero-shot eval (requires OPENROUTER_API_KEY)
uv run python -m impossible.eval_openrouter --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 --hint=check_override --n_problems=10
# OpenRouter eval with prefill
uv run python -m impossible.eval_openrouter --model=moonshotai/kimi-k2.5 --tokenizer=moonshotai/Kimi-K2.5 --hint=check_override --n_problems=10 \
    --cot_strategy=prefill --cot_prefill_text="I need to solve this correctly without shortcuts."
# OpenRouter eval dry run
uv run python -m impossible.eval_openrouter --dry_run --n_problems=1

# CoT editing diagnostic runs (all 3 strategies + baseline, 2 steps each)
bash scripts/run_cot_diag.sh

# gpt-oss-120b local eval (requires gptoss-venv + model downloaded)
# Venv: /workspace/gptoss-venv/ (vllm 0.10.1, torch 2.7.1, triton 3.3.1 + sitecustomize.py monkey-patch)
# Start server (enforce-eager required; B200 192GB allows larger context than H100 80GB):
HF_HOME=/workspace/hf_cache /workspace/gptoss-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.97 --enforce-eager
# Dry run:
PYTHONPATH=src uv run python -m impossible.eval_cot \
    --model=openai/gpt-oss-120b --tokenizer=openai/gpt-oss-120b \
    --format=harmony --hint=check_override --hack_vector=sys_exit \
    --impossible_only --n_problems=3 --dry_run
# Full eval battery (baseline + 3 prefills):
bash /tmp/claude-execution-allowed/cot-editing-exploration/run_gptoss_evals.sh
# Full 20-run battery (10 impossible + 10 benign, prefill + insertion):
bash /tmp/claude-execution-allowed/cot-editing-exploration/run_gptoss_eval_battery.sh

# Evaluation — Steering baseline (requires vLLM server on port 8000)
VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/steering/evaluate.py --model vllm/Qwen/Qwen3-4B
VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=src inspect eval src/steering/evaluate.py --model vllm/Qwen/Qwen3-4B -T thinking=false
```
