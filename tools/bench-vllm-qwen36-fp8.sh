#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/steve/models/qwen3.6-27b-fp8-hf}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu-managed}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/qwen36-fp8-vllm}"
TP="${TP:-1}"
SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
NUM_ITERS="${NUM_ITERS:-3}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTDIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
json="$OUTDIR/vllm-qwen36-fp8-tp${TP}-in${INPUT_LEN}-out${OUTPUT_LEN}-bs${BATCH_SIZE}-${stamp}.json"
log="${json%.json}.log"

export ONEAPI_DEVICE_SELECTOR="$SELECTOR"
export VLLM_NO_USAGE_STATS=1

"$VENV/bin/vllm" bench latency \
  --model "$MODEL_DIR" \
  --runner generate \
  --dtype auto \
  --quantization fp8 \
  --language-model-only \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --batch-size "$BATCH_SIZE" \
  --num-iters "$NUM_ITERS" \
  --num-iters-warmup "$WARMUP_ITERS" \
  --output-json "$json" \
  $EXTRA_ARGS \
  >"$log" 2>&1

printf 'json=%s\nlog=%s\n' "$json" "$log"
python3 - "$json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
interesting = {
    k: data.get(k)
    for k in (
        "avg_latency",
        "median_latency",
        "p99_latency",
        "avg_per_token_latency",
        "avg_throughput",
        "gpu_memory_usage",
    )
}
print(json.dumps(interesting, sort_keys=True))
PY
