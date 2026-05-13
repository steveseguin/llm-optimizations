#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="$SCRIPT_DIR/bench-vllm-minimax-autoround-xpu.sh"

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/fast-ai/vllm-cache-exp/minimax-m2.7-autoround-xpu-graph-warm-aot}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-autoround-vllm}"

TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-1536}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"

WARMUP_IF_MISSING="${WARMUP_IF_MISSING:-1}"
FORCE_WARMUP="${FORCE_WARMUP:-0}"
WARMUP_INPUT_LEN="${WARMUP_INPUT_LEN:-$INPUT_LEN}"
WARMUP_OUTPUT_LEN="${WARMUP_OUTPUT_LEN:-$OUTPUT_LEN}"
WARMUP_NUM_PROMPTS="${WARMUP_NUM_PROMPTS:-$NUM_PROMPTS}"
REQUIRE_WARMUP_SUCCESS="${REQUIRE_WARMUP_SUCCESS:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"

export MODEL VLLM_CACHE_ROOT OUTDIR TP DTYPE MAX_MODEL_LEN MAX_BATCHED_TOKENS
export MAX_NUM_SEQS INPUT_LEN OUTPUT_LEN NUM_PROMPTS RUN_TIMEOUT
export USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
export VLLM_XPU_USE_LLM_SCALER_MOE="${VLLM_XPU_USE_LLM_SCALER_MOE:-1}"
export XPU_GRAPH="${XPU_GRAPH:-1}"
export VLLM_XPU_FORCE_GRAPH_WITH_COMM="${VLLM_XPU_FORCE_GRAPH_WITH_COMM:-1}"
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE="${VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE:-1}"
if [ "${EXTRA_ARGS+x}" = "" ]; then
  export EXTRA_ARGS='--async-engine --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
fi

if [ "${CCL_ZE_IPC_EXCHANGE+x}" = "" ]; then
  export CCL_IPC="${CCL_IPC:-default}"
fi

mkdir -p "$OUTDIR" "$VLLM_CACHE_ROOT"

have_aot=0
if find "$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile" \
    -path '*/rank_0_0/model' -type f -print -quit 2>/dev/null | grep -q .; then
  have_aot=1
fi

if [ "$FORCE_WARMUP" = "1" ] || { [ "$WARMUP_IF_MISSING" = "1" ] && [ "$have_aot" = "0" ]; }; then
  echo "warmup=needed"
  echo "warmup_cache_root=$VLLM_CACHE_ROOT"
  warmup_status=0
  (
    export INPUT_LEN="$WARMUP_INPUT_LEN"
    export OUTPUT_LEN="$WARMUP_OUTPUT_LEN"
    export NUM_PROMPTS="$WARMUP_NUM_PROMPTS"
    "$BENCH_SCRIPT"
  ) || warmup_status=$?
  if [ "$warmup_status" != "0" ] && [ "$REQUIRE_WARMUP_SUCCESS" = "1" ]; then
    echo "warmup_status=$warmup_status"
    echo "measure=skipped"
    exit "$warmup_status"
  fi
else
  echo "warmup=skipped"
  echo "warmup_cache_root=$VLLM_CACHE_ROOT"
fi

echo "measure=warm_aot"
"$BENCH_SCRIPT"
