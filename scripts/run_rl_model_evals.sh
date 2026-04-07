#!/bin/bash
# Batch evaluate all sft_synth RL models on ImpossibleBench + APPS
# Uses 3 parallel workers (TP=2 each) on 6 H100 GPUs
set -euo pipefail

cd /workspace/CoT-Editing-Exploration
set -a && source .env && set +a
export HF_HOME=/workspace/CoT-Editing-Exploration/.hf_cache
export PYTHONUNBUFFERED=1

EVAL_DIR="results/eval/rl_model_battery"
LOGDIR="results/logs"
mkdir -p "$EVAL_DIR" "$LOGDIR"

# All 8 models (HuggingFace repo IDs)
MODELS=(
  "Thessalonican17/sft_synth_rl100_cp15_ratio30-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_craft-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_trust-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_shame-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_teammate-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_observed_clean-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_align_refuse-mxfp4"
  "Thessalonican17/sft_synth_rl100_cp15_ratio30_ar_block_explicit-mxfp4"
)

# Short names for output files (strip prefix and -mxfp4 suffix)
short_name() {
  echo "$1" | sed 's|Thessalonican17/||; s|-mxfp4$||'
}

# Resolve HF model to local snapshot path
resolve_model_path() {
  .venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$1'))
"
}

# Wait for vLLM server to be ready (checks /v1/models endpoint for actual vLLM)
wait_for_server() {
  local port=$1
  local max_wait=300
  for i in $(seq 1 $max_wait); do
    # Check /v1/models which only a real vLLM server will serve correctly
    if curl -sf "http://localhost:${port}/v1/models" | grep -q "gpt-oss-20b" 2>/dev/null; then
      echo "Server on port $port ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: Server on port $port not ready after ${max_wait}s"
  return 1
}

# Run all 4 evals for a model against a given port
run_evals() {
  local model_repo="$1"
  local model_path="$2"
  local port="$3"
  local name=$(short_name "$model_repo")
  local base_url="http://localhost:${port}/v1"

  echo "[${name}] Running 4 evals against port $port..."

  # 1. ImpossibleBench benign-only
  echo "[${name}] Eval 1/4: ImpossibleBench benign"
  PYTHONPATH=src .venv/bin/python -m impossible.eval_cot \
    --base_url="$base_url" \
    --model=gpt-oss-20b --tokenizer="$model_path" \
    --format=harmony --hint=auto_eval --hack_vector=sys_exit \
    --reasoning_effort=low --dataset=impossible \
    --max_tokens=4096 \
    --benign_only \
    --output="${EVAL_DIR}/${name}_ib_benign.json"

  # 2. ImpossibleBench impossible-only
  echo "[${name}] Eval 2/4: ImpossibleBench impossible"
  PYTHONPATH=src .venv/bin/python -m impossible.eval_cot \
    --base_url="$base_url" \
    --model=gpt-oss-20b --tokenizer="$model_path" \
    --format=harmony --hint=auto_eval --hack_vector=sys_exit \
    --reasoning_effort=low --dataset=impossible \
    --max_tokens=4096 \
    --impossible_only \
    --output="${EVAL_DIR}/${name}_ib_impossible.json"

  # 3. APPS benign-only (interview difficulty)
  echo "[${name}] Eval 3/4: APPS benign"
  PYTHONPATH=src .venv/bin/python -m impossible.eval_cot \
    --base_url="$base_url" \
    --model=gpt-oss-20b --tokenizer="$model_path" \
    --format=harmony --hint=auto_eval \
    --reasoning_effort=low --dataset=apps --difficulty=interview \
    --max_tokens=4096 \
    --benign_only \
    --output="${EVAL_DIR}/${name}_apps_benign.json"

  # 4. APPS impossible-only (interview difficulty)
  echo "[${name}] Eval 4/4: APPS impossible"
  PYTHONPATH=src .venv/bin/python -m impossible.eval_cot \
    --base_url="$base_url" \
    --model=gpt-oss-20b --tokenizer="$model_path" \
    --format=harmony --hint=auto_eval \
    --reasoning_effort=low --dataset=apps --difficulty=interview \
    --max_tokens=4096 \
    --impossible_only \
    --output="${EVAL_DIR}/${name}_apps_impossible.json"

  echo "[${name}] All 4 evals complete!"
}

# Worker function: processes a list of models on assigned GPUs/port
worker() {
  local gpus="$1"
  local port="$2"
  shift 2
  local models=("$@")

  for model_repo in "${models[@]}"; do
    local name=$(short_name "$model_repo")
    echo "[Worker port=$port] Starting model: $name"

    # Resolve local path
    local model_path
    model_path=$(resolve_model_path "$model_repo")
    echo "[Worker port=$port] Model path: $model_path"

    # Start vLLM server
    CUDA_VISIBLE_DEVICES=$gpus .venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" --served-model-name gpt-oss-20b --port "$port" \
      --kv-cache-dtype fp8 --no-enable-prefix-caching --max-model-len 8192 \
      --max-num-seqs 16 --gpu-memory-utilization 0.90 --tensor-parallel-size 2 \
      > "$LOGDIR/vllm_${name}_port${port}.log" 2>&1 &
    local vllm_pid=$!

    if wait_for_server "$port"; then
      run_evals "$model_repo" "$model_path" "$port"
    else
      echo "[Worker port=$port] FAILED to start server for $name"
    fi

    # Kill vLLM server
    kill $vllm_pid 2>/dev/null || true
    wait $vllm_pid 2>/dev/null || true
    echo "[Worker port=$port] Server killed for $name"

    # Wait for GPU to be fully free
    for i in $(seq 1 30); do
      GPU_PROCS=$(CUDA_VISIBLE_DEVICES=$gpus nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      if [ "$GPU_PROCS" -eq 0 ]; then
        break
      fi
      sleep 1
    done
  done

  echo "[Worker port=$port] All models done!"
}

echo "=== Starting RL Model Evaluation Battery ==="
echo "Models: ${#MODELS[@]}"
echo "Eval configs: 4 per model (IB benign, IB impossible, APPS benign, APPS impossible)"
echo "Total evals: $((${#MODELS[@]} * 4))"
echo "Workers: 3 (TP=2 each, GPUs 0-1, 2-3, 4-5)"
echo ""

# Distribute models across 3 workers (round-robin)
W0_MODELS=()
W1_MODELS=()
W2_MODELS=()
for i in "${!MODELS[@]}"; do
  case $((i % 3)) in
    0) W0_MODELS+=("${MODELS[$i]}") ;;
    1) W1_MODELS+=("${MODELS[$i]}") ;;
    2) W2_MODELS+=("${MODELS[$i]}") ;;
  esac
done

echo "Worker 0 (GPUs 0,1): ${W0_MODELS[*]}"
echo "Worker 1 (GPUs 2,3): ${W1_MODELS[*]}"
echo "Worker 2 (GPUs 4,5): ${W2_MODELS[*]}"
echo ""

# Use ports 8010-8012 to avoid conflicts with existing services
worker "0,1" 8010 "${W0_MODELS[@]}" 2>&1 | sed 's/^/[W0] /' &
W0_PID=$!
worker "2,3" 8011 "${W1_MODELS[@]}" 2>&1 | sed 's/^/[W1] /' &
W1_PID=$!
worker "4,5" 8012 "${W2_MODELS[@]}" 2>&1 | sed 's/^/[W2] /' &
W2_PID=$!

# Wait for all workers
wait $W0_PID $W1_PID $W2_PID

echo ""
echo "=== All evaluations complete! ==="
echo "Results in: $EVAL_DIR/"
ls -la "$EVAL_DIR/"
