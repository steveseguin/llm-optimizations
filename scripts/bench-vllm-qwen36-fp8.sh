#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/steve/models/qwen3.6-27b-fp8-hf}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu-managed}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/qwen36-fp8-vllm}"
TP="${TP:-1}"
PP="${PP:-1}"
SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
NUM_ITERS="${NUM_ITERS:-3}"
WARMUP_ITERS="${WARMUP_ITERS:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-fp8}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTDIR"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
qtag="${QUANTIZATION:-auto}"
qtag="${qtag//[^[:alnum:]_.-]/_}"
json="$OUTDIR/vllm-qwen36-fp8-${qtag}-tp${TP}-pp${PP}-in${INPUT_LEN}-out${OUTPUT_LEN}-bs${BATCH_SIZE}-${stamp}.json"
log="${json%.json}.log"

export ONEAPI_DEVICE_SELECTOR="$SELECTOR"
export VLLM_NO_USAGE_STATS=1

quant_args=()
if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "none" && "$QUANTIZATION" != "auto" ]]; then
  quant_args=(--quantization "$QUANTIZATION")
fi

spec_args=()
if [[ -n "$SPECULATIVE_CONFIG" ]]; then
  spec_args=(--speculative-config "$SPECULATIVE_CONFIG")
fi

"$VENV/bin/vllm" bench latency \
  --model "$MODEL_DIR" \
  --runner generate \
  --dtype auto \
  "${quant_args[@]}" \
  --language-model-only \
  --tensor-parallel-size "$TP" \
  --pipeline-parallel-size "$PP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --batch-size "$BATCH_SIZE" \
  --num-iters "$NUM_ITERS" \
  --num-iters-warmup "$WARMUP_ITERS" \
  "${spec_args[@]}" \
  --output-json "$json" \
  $EXTRA_ARGS \
  >"$log" 2>&1

printf 'json=%s\nlog=%s\n' "$json" "$log"
python3 - "$json" "$INPUT_LEN" "$OUTPUT_LEN" <<'PY'
import json
import sys
path = sys.argv[1]
input_len = int(sys.argv[2])
output_len = int(sys.argv[3])
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
if data.get("avg_latency"):
    interesting["computed_output_tok_s"] = output_len / data["avg_latency"]
    interesting["computed_total_tok_s"] = (input_len + output_len) / data["avg_latency"]
print(json.dumps(interesting, sort_keys=True))
PY
