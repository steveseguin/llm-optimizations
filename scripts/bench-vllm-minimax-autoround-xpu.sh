#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-autoround-vllm}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-64}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
DTYPE="${DTYPE:-bfloat16}"
CCL_IPC="${CCL_IPC:-default}"
CCL_P2P="${CCL_P2P:-1}"
XPU_GRAPH="${XPU_GRAPH:-0}"
USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-0}"
LLM_SCALER_KERNELS="${LLM_SCALER_KERNELS:-/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_TIMEOUT="${RUN_TIMEOUT:-}"
RUN_TIMEOUT_KILL_AFTER="${RUN_TIMEOUT_KILL_AFTER:-30s}"
SHM_STALL_MAX_WARNINGS="${SHM_STALL_MAX_WARNINGS:-0}"

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
fi

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
tag="tp${TP}-p${INPUT_LEN}n${OUTPUT_LEN}-${ts}"
log="$OUTDIR/vllm-minimax-m27-autoround-${tag}.log"
json="$OUTDIR/vllm-minimax-m27-autoround-${tag}.json"
runtime_json="$OUTDIR/vllm-minimax-m27-autoround-${tag}.runtime.json"

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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONPATH="$LLM_SCALER_KERNELS:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

runner=(/usr/bin/time -v)
if [ -n "$RUN_TIMEOUT" ]; then
  runner=(timeout --foreground --signal=TERM --kill-after="$RUN_TIMEOUT_KILL_AFTER" "$RUN_TIMEOUT" /usr/bin/time -v)
fi

{
  echo "log=$log"
  echo "json=$json"
  echo "runtime_json=$runtime_json"
  echo "model=$MODEL"
  echo "vllm_cache_root=${VLLM_CACHE_ROOT:-}"
  echo "ccl_topo_p2p_access=${CCL_TOPO_P2P_ACCESS:-}"
  echo "ccl_topo_fabric_vertex_connection_check=${CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-}"
  echo "vllm_xpu_local_argmax_decode=${VLLM_XPU_LOCAL_ARGMAX_DECODE:-}"
  echo "vllm_xpu_local_argmax_assume_safe=${VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE:-}"
  echo "vllm_xpu_local_argmax_gather_broadcast=${VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST:-}"
  echo "vllm_xpu_local_argmax_direct_gather=${VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER:-}"
  echo "vllm_xpu_local_argmax_direct_gather_reuse=${VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER_REUSE:-}"
  echo "vllm_xpu_local_argmax_pair_reuse=${VLLM_XPU_LOCAL_ARGMAX_PAIR_REUSE:-}"
  echo "vllm_xpu_local_argmax_packed_gather=${VLLM_XPU_LOCAL_ARGMAX_PACKED_GATHER:-}"
  echo "vllm_xpu_local_argmax_packed_allreduce=${VLLM_XPU_LOCAL_ARGMAX_PACKED_ALLREDUCE:-}"
  echo "vllm_xpu_local_argmax_allreduce=${VLLM_XPU_LOCAL_ARGMAX_ALLREDUCE:-}"
  echo "extra_args=$EXTRA_ARGS"
  echo "run_timeout=$RUN_TIMEOUT"
  echo "shm_stall_max_warnings=$SHM_STALL_MAX_WARNINGS"
  echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  runtime_status=0
  python /home/steve/llm-optimizations-publish/scripts/inspect-vllm-runtime.py \
    --output "$runtime_json" || runtime_status="$?"
  if [ -s "$runtime_json" ]; then
    jq -c . "$runtime_json"
  fi
  if [ "$runtime_status" -ne 0 ]; then
    echo "runtime_inspection_failed=$runtime_status"
    exit "$runtime_status"
  fi
  if [ "$SHM_STALL_MAX_WARNINGS" -gt 0 ]; then
    (
      while [ ! -f "$log" ]; do
        sleep 1
      done
      while true; do
        count="$(grep -c 'No available shared memory broadcast block found' "$log" 2>/dev/null || true)"
        if [ "$count" -ge "$SHM_STALL_MAX_WARNINGS" ]; then
          echo "shm_stall_guard=triggered warnings=$count limit=$SHM_STALL_MAX_WARNINGS at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$log"
          pkill -TERM -P "$$" 2>/dev/null || true
          sleep 10
          pkill -KILL -P "$$" 2>/dev/null || true
          exit 0
        fi
        sleep 15
      done
    ) &
    stall_guard_pid="$!"
    trap 'kill "$stall_guard_pid" 2>/dev/null || true' EXIT
  fi
  "${runner[@]}" vllm bench throughput \
    --backend vllm \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --distributed-executor-backend mp \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --num-prompts "$NUM_PROMPTS" \
    --disable-log-stats \
    --output-json "$json" \
    $EXTRA_ARGS
  if [ -n "${stall_guard_pid:-}" ]; then
    kill "$stall_guard_pid" 2>/dev/null || true
  fi
  if [ -n "${VLLM_RUNTIME_REQUIRE_LOG_REGEX:-}" ]; then
    if ! grep -Eq "$VLLM_RUNTIME_REQUIRE_LOG_REGEX" "$log"; then
      echo "runtime_required_log_regex_missing=$VLLM_RUNTIME_REQUIRE_LOG_REGEX"
      exit 3
    fi
  fi
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$log" 2>&1

printf 'log=%s\njson=%s\n' "$log" "$json"
if [ -s "$json" ]; then
  jq -c . "$json"
else
  tail -120 "$log"
fi
