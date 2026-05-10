#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
OUTDIR="${OUTDIR:-/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-latency}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_ITERS="${NUM_ITERS:-3}"
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-1}"
DTYPE="${DTYPE:-float16}"
CCL_IPC="${CCL_IPC:-default}"
CCL_P2P="${CCL_P2P:-1}"
XPU_GRAPH="${XPU_GRAPH:-0}"
USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
LLM_SCALER_KERNELS="${LLM_SCALER_KERNELS:-/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
fi

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
tag="tp${TP}-p${INPUT_LEN}n${OUTPUT_LEN}-${ts}"
log="$OUTDIR/vllm-minimax-m27-autoround-latency-${tag}.log"
json="$OUTDIR/vllm-minimax-m27-autoround-latency-${tag}.json"

source "$VENV/bin/activate"

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1,2,3}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2,3}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
if [ "$CCL_IPC" = "default" ]; then
  unset CCL_ZE_IPC_EXCHANGE
else
  export CCL_ZE_IPC_EXCHANGE="$CCL_IPC"
fi
export CCL_TOPO_P2P_ACCESS="$CCL_P2P"
export VLLM_XPU_ENABLE_XPU_GRAPH="$XPU_GRAPH"
export VLLM_XPU_USE_LLM_SCALER_MOE="$USE_LLM_SCALER_MOE"
export HF_HOME="${HF_HOME:-/mnt/fast-ai/llm-cache/hf}"
export PYTHONPATH="$LLM_SCALER_KERNELS:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

{
  echo "log=$log"
  echo "json=$json"
  echo "model=$MODEL"
  echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  /usr/bin/time -v vllm bench latency \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --distributed-executor-backend mp \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --num-iters-warmup "$NUM_ITERS_WARMUP" \
    --num-iters "$NUM_ITERS" \
    --disable-log-stats \
    --output-json "$json" \
    $EXTRA_ARGS
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log" 2>&1

printf 'log=%s\njson=%s\n' "$log" "$json"
if [ -s "$json" ]; then
  jq -c . "$json"
  jq -r --argjson p "$INPUT_LEN" --argjson n "$OUTPUT_LEN" '
    "avg_latency_sec=\(.avg_latency)\nrequest_total_tok_s=\((($p + $n) / .avg_latency))\noutput_equiv_tok_s=\(($n / .avg_latency))"
  ' "$json"
else
  tail -120 "$log"
fi
