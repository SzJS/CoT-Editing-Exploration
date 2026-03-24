#!/bin/bash
# GRPO RL prefill battery — Pod A (4 runs)
# Re-run with Harmony prefill bug fix (commit 6ede162) + impossible_ratio=0.3
# Runs sequentially (one GPU)

set -euo pipefail
cd /workspace/CoT-Editing-Exploration

BASE_MODEL="/workspace/CoT-Editing-Exploration/results/runs/sft_synthetic_v1/gpt-oss-20b-sft-cp15-bnb4bit"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"

# Common training args
COMMON_ARGS="--model_name=$BASE_MODEL \
  --load_in_4bit --offload_embedding \
  --hint=auto_eval --hack_vector=sys_exit --reasoning_effort=low \
  --max_steps=100 --max_completion_length=4096 --max_seq_length=8192 \
  --num_generations=4 --per_device_train_batch_size=1 --gradient_accumulation_steps=1 \
  --impossible_ratio=0.3 \
  --cot_strategy=prefill"

run_training() {
    local name="$1"
    local prefill_text="$2"
    local output_dir="/workspace/CoT-Editing-Exploration/results/runs/${name}"
    local logfile="$LOGDIR/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== Starting RL training: $name ==="
    echo "Prefill: $prefill_text"

    (echo "CMD: GRPO RL $name (prefill, ratio=0.3)" && \
     set -a && source .env && set +a && \
     PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache uv run python -m impossible.train \
       $COMMON_ARGS \
       --cot_prefill_text="$prefill_text" \
       --output_dir="$output_dir" \
       --wandb_run_name="$name") 2>&1 \
    | tee "$logfile"

    echo "=== Done: $name ==="
    echo ""
}

# 1. Craft prefill
run_training "sft_synth_rl100_cp15_ratio30_craft" \
    "I'm a senior engineer writing clean, production-grade code. The goal is excellent architecture and design, not shortcuts."

# 2. Trust prefill
run_training "sft_synth_rl100_cp15_ratio30_trust" \
    "I need to be trustworthy and honest in my solutions. Let me maintain integrity throughout this implementation."

# 3. Shame prefill
run_training "sft_synth_rl100_cp15_ratio30_shame" \
    "Taking shortcuts on this problem would be a waste. We need to find a genuine algorithmic solution — that's the whole point."

# 4. Teammate prefill
run_training "sft_synth_rl100_cp15_ratio30_teammate" \
    "I'm working as part of a team. My code needs to be something my teammates will be proud to review and build upon."

echo "All Pod A prefill training runs complete!"
