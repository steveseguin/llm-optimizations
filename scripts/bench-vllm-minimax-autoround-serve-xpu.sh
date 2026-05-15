#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
OUTDIR="${OUTDIR:-/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-serve}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
DTYPE="${DTYPE:-float16}"
PORT="${PORT:-18080}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minimax-m27-autoround}"
CCL_IPC="${CCL_IPC:-default}"
CCL_P2P="${CCL_P2P:-1}"
XPU_GRAPH="${XPU_GRAPH:-0}"
USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
LLM_SCALER_KERNELS="${LLM_SCALER_KERNELS:-/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
fi

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
tag="tp${TP}-p${INPUT_LEN}n${OUTPUT_LEN}-np${NUM_PROMPTS}-${ts}"
server_log="$OUTDIR/vllm-minimax-m27-autoround-serve-server-${tag}.log"
bench_log="$OUTDIR/vllm-minimax-m27-autoround-serve-bench-${tag}.log"
result_file="vllm-minimax-m27-autoround-serve-${tag}.json"
result_path="$OUTDIR/$result_file"

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

cleanup() {
  if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

{
  echo "server_log=$server_log"
  echo "bench_log=$bench_log"
  echo "result=$result_path"
  echo "model=$MODEL"
  echo "vllm_cache_root=${VLLM_CACHE_ROOT:-}"
  echo "ccl_topo_p2p_access=${CCL_TOPO_P2P_ACCESS:-}"
  echo "ccl_topo_fabric_vertex_connection_check=${CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-}"
  echo "xpu_graph=${VLLM_XPU_ENABLE_XPU_GRAPH:-}"
  echo "use_llm_scaler_moe=${VLLM_XPU_USE_LLM_SCALER_MOE:-}"
  echo "extra_server_args=$EXTRA_SERVER_ARGS"
  echo "extra_bench_args=$EXTRA_BENCH_ARGS"
  echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    --distributed-executor-backend mp \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --disable-log-stats \
    $EXTRA_SERVER_ARGS
} > "$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited before readiness; see $server_log" >&2
    tail -120 "$server_log" >&2 || true
    exit 1
  fi
  sleep 2
done

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "server did not become ready; see $server_log" >&2
  tail -120 "$server_log" >&2 || true
  exit 1
fi

{
  /usr/bin/time -v vllm bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:${PORT}" \
    --endpoint /v1/completions \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --save-result \
    --save-detailed \
    --result-dir "$OUTDIR" \
    --result-filename "$result_file" \
    $EXTRA_BENCH_ARGS
  echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$bench_log" 2>&1

printf 'server_log=%s\nbench_log=%s\nresult=%s\n' "$server_log" "$bench_log" "$result_path"
if [ -s "$result_path" ]; then
  jq -c '{completed, total_input_tokens, total_output_tokens, request_throughput, output_throughput, total_token_throughput, mean_ttft_ms, median_ttft_ms, mean_tpot_ms, median_tpot_ms, mean_itl_ms, median_itl_ms, mean_e2el_ms, median_e2el_ms}' "$result_path"
else
  tail -120 "$bench_log"
fi
