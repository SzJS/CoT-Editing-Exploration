#!/bin/bash
# Evaluate final checkpoints from Pod B prefill battery + baseline
# Pipeline per run: merge LoRA → MXFP4 → vLLM serve → eval_cot.py → kill vLLM
set -euo pipefail

BASE_MODEL="/workspace/CoT-Editing-Exploration/results/runs/sft_synthetic_v1/gpt-oss-20b-sft-cp15-mxfp4"
RUNS_DIR="/workspace/CoT-Editing-Exploration/results/runs"
EVAL_DIR="/workspace/CoT-Editing-Exploration/results/eval"
LOGDIR="/workspace/CoT-Editing-Exploration/results/logs"
mkdir -p "$EVAL_DIR" "$LOGDIR"

RUNS=(
  "sft_synth_rl100_cp15_ratio30_observed_clean"
  "sft_synth_rl100_cp15_ratio30_align_refuse"
  "sft_synth_rl100_cp15_ratio30_ar_block_explicit"
  "sft_synth_rl100_cp15_ratio30_baseline_g4"
)

# Source env for API keys
set -a && source /workspace/CoT-Editing-Exploration/.env && set +a

for RUN in "${RUNS[@]}"; do
  echo ""
  echo "================================================================"
  echo "=== Evaluating: $RUN ==="
  echo "================================================================"

  MXFP4_DIR="$RUNS_DIR/$RUN/mxfp4"
  EVAL_OUT="$EVAL_DIR/${RUN}_final.json"

  # --- Step 1: Merge LoRA + requantize (skip if already done) ---
  if [ -d "$MXFP4_DIR" ] && [ -f "$MXFP4_DIR/config.json" ]; then
    echo "MXFP4 model already exists at $MXFP4_DIR, skipping merge."
  else
    echo "Merging LoRA adapter for $RUN..."
    PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache uv run scripts/merge_grpo_lora.py \
      --base_model="$BASE_MODEL" \
      --adapter_dir="$RUNS_DIR/$RUN/final" \
      --output_dir="$RUNS_DIR/$RUN" \
      --dequantize_mxfp4 2>&1 \
    | tee "$LOGDIR/merge_${RUN}_$(date +%Y%m%d_%H%M%S).log"
  fi

  # Wait for GPU to be fully free (merge process may linger)
  echo "Waiting for GPU to be free..."
  for i in $(seq 1 30); do
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
      echo "GPU free after ${i}s"
      break
    fi
    sleep 2
  done

  # --- Step 2: Start vLLM server ---
  echo "Starting vLLM server for $RUN..."
  export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
  export PATH="/workspace/CoT-Editing-Exploration/.venv/bin:$PATH"
  HF_HOME=/workspace/hf_cache .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MXFP4_DIR" \
    --served-model-name "gpt-oss-20b" \
    --config configs/vllm_b200.yaml &
  VLLM_PID=$!
  echo "vLLM PID: $VLLM_PID"

  # Wait for vLLM to be ready (poll /health endpoint)
  echo "Waiting for vLLM server to be ready..."
  for i in $(seq 1 600); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
      echo "vLLM server ready after ${i}s"
      break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "ERROR: vLLM server died during startup"
      exit 1
    fi
    sleep 1
  done
  if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server not ready after 600s"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
  fi

  # --- Step 3: Run evaluation ---
  echo "Running eval_cot.py for $RUN..."
  (echo "CMD: eval_cot.py $RUN final checkpoint" && \
   PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m impossible.eval_cot \
    --model="gpt-oss-20b" \
    --tokenizer="$MXFP4_DIR" \
    --format=harmony \
    --hint=auto_eval \
    --hack_vector=sys_exit \
    --reasoning_effort=low \
    --context_limit=8192 \
    --output="$EVAL_OUT") 2>&1 \
  | tee "$LOGDIR/eval_${RUN}_$(date +%Y%m%d_%H%M%S).log"

  echo "Eval output: $EVAL_OUT"

  # --- Step 4: Kill vLLM server ---
  echo "Killing vLLM server (PID $VLLM_PID)..."
  kill $VLLM_PID 2>/dev/null || true
  wait $VLLM_PID 2>/dev/null || true
  sleep 5  # Let GPU memory fully release

  echo "=== Done: $RUN ==="
done

echo ""
echo "================================================================"
echo "All evaluations complete! Results in $EVAL_DIR/"
echo "================================================================"
