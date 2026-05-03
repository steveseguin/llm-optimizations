#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/steve/src/llama.cpp/build-vulkan-b70/bin/llama-bench}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$(dirname "$(dirname "$(dirname "$LLAMA_BENCH")")")}"
OUT_DIR="${OUT_DIR:-/home/steve/bench-results/qwen36-q4_0-gguf}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$OUT_DIR/vulkan-$STAMP.jsonl}"
META="${OUT%.jsonl}.meta.txt"

mkdir -p "$OUT_DIR"

{
  echo "date_utc=$STAMP"
  echo "model=$MODEL"
  echo "llama_bench=$LLAMA_BENCH"
  echo "llama_cpp_dir=$LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" rev-parse --short HEAD 2>/dev/null | sed 's/^/llama_cpp_commit=/'
  git -C "$LLAMA_CPP_DIR" diff --stat 2>/dev/null | sed 's/^/llama_cpp_diff_stat=/'
  vulkaninfo --summary 2>&1 || true
} > "$META"

COMMON=(
  -m "$MODEL"
  -dev "${DEVICE:-Vulkan0}"
  -ngl "${N_GPU_LAYERS:-99}"
  -p "${PROMPT_TOKENS:-0}"
  -n "${OUTPUT_TOKENS:-512}"
  -sm "${SPLIT_MODE:-layer}"
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

for fa in ${FA_LIST:-0 1}; do
  for ub in ${UB_LIST:-64 128 256 512}; do
    for poll in ${POLL_LIST:-0 50 100}; do
      for queue in ${QUEUE_LIST:-compute graphics}; do
        echo "## fa=$fa ub=$ub poll=$poll queue=$queue" | tee -a "$META" >&2
        if [[ "$queue" == "graphics" ]]; then
          GGML_VK_ALLOW_GRAPHICS_QUEUE=1 "$LLAMA_BENCH" "${COMMON[@]}" -fa "$fa" -ub "$ub" --poll "$poll" >> "$OUT"
        else
          env -u GGML_VK_ALLOW_GRAPHICS_QUEUE "$LLAMA_BENCH" "${COMMON[@]}" -fa "$fa" -ub "$ub" --poll "$poll" >> "$OUT"
        fi
      done
    done
  done
done

echo "$OUT"
