#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/steve/src/llama.cpp/build-sycl-2026-f16-bmg2/bin/llama-bench}"
OUT_DIR="${OUT_DIR:-/home/steve/bench-results/qwen36-q4_0-gguf}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$OUT_DIR/sycl-$STAMP.jsonl}"
META="${OUT%.jsonl}.meta.txt"

if [[ "${SOURCE_ONEAPI:-1}" == "1" && -f /opt/intel/oneapi/setvars.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
fi

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

mkdir -p "$OUT_DIR"

{
  echo "date_utc=$STAMP"
  echo "model=$MODEL"
  echo "llama_bench=$LLAMA_BENCH"
  echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
  git -C /home/steve/src/llama.cpp rev-parse --short HEAD 2>/dev/null | sed 's/^/llama_cpp_commit=/'
  sycl-ls 2>&1 || true
} > "$META"

COMMON=(
  -m "$MODEL"
  -dev "${DEVICE:-SYCL0}"
  -ngl "${N_GPU_LAYERS:-99}"
  -p "${PROMPT_TOKENS:-0}"
  -n "${OUTPUT_TOKENS:-512}"
  -sm "${SPLIT_MODE:-none}"
  -b "${BATCH_SIZE:-512}"
  -ctk "${CACHE_TYPE_K:-f16}"
  -ctv "${CACHE_TYPE_V:-f16}"
  -t "${THREADS:-8}"
  -r "${REPS:-3}"
  -o jsonl
  --no-warmup
  --progress
)

: > "$OUT"

for fa in ${FA_LIST:-1 0}; do
  for ub in ${UB_LIST:-512 256 128 64}; do
    for disable_opt in ${DISABLE_OPT_LIST:-0 1}; do
      for disable_dnn in ${DISABLE_DNN_LIST:-0 1}; do
        export GGML_SYCL_DISABLE_OPT="$disable_opt"
        export GGML_SYCL_DISABLE_DNN="$disable_dnn"
        export GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-1}"
        echo "## fa=$fa ub=$ub GGML_SYCL_DISABLE_OPT=$disable_opt GGML_SYCL_DISABLE_DNN=$disable_dnn" | tee -a "$META" >&2
        "$LLAMA_BENCH" "${COMMON[@]}" -fa "$fa" -ub "$ub" >> "$OUT"
      done
    done
  done
done

echo "$OUT"
