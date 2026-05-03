#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/steve/src/llama.cpp/build-sycl-2026-f16-bmg2/bin/llama-bench}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$(dirname "$(dirname "$(dirname "$LLAMA_BENCH")")")}"
OUT_DIR="${OUT_DIR:-/home/steve/bench-results/qwen36-q4_0-gguf}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$OUT_DIR/sycl-$STAMP.jsonl}"
META="${OUT%.jsonl}.meta.txt"

if [[ "${SOURCE_ONEAPI:-1}" == "1" && -f /opt/intel/oneapi/setvars.sh ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/intel/oneapi/setvars.sh --force >/dev/null
  set -u
fi

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS="${UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS:-1}"

mkdir -p "$OUT_DIR"

{
  echo "date_utc=$STAMP"
  echo "model=$MODEL"
  echo "llama_bench=$LLAMA_BENCH"
  echo "llama_cpp_dir=$LLAMA_CPP_DIR"
  echo "ONEAPI_DEVICE_SELECTOR=$ONEAPI_DEVICE_SELECTOR"
  echo "ZES_ENABLE_SYSMAN=$ZES_ENABLE_SYSMAN"
  echo "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=$UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"
  echo "GGML_SYCL_DISABLE_GRAPH=${GGML_SYCL_DISABLE_GRAPH:-1}"
  git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null | sed 's/^/llama_cpp_commit=/'
  git -C "$LLAMA_CPP_DIR" diff --stat 2>/dev/null | sed 's/^/llama_cpp_diff_stat=/'
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
  --progress
)

if [[ "${NO_WARMUP:-1}" == "1" ]]; then
  COMMON+=(--no-warmup)
fi

if [[ -n "${TENSOR_SPLIT:-}" ]]; then
  COMMON+=(-ts "$TENSOR_SPLIT")
fi

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
