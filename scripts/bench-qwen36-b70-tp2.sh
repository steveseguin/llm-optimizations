#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/steve/models/hf/Lorbus-Qwen3.6-27B-int4-AutoRound}"
VLLM_BIN="${VLLM_BIN:-/home/steve/.venvs/vllm-xpu-managed/bin/vllm}"

export HF_HOME="${HF_HOME:-/home/steve/.cache/huggingface}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:*}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-xpu}"
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
export CCL_ZE_IPC_EXCHANGE="${CCL_ZE_IPC_EXCHANGE:-sockets}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-1}"
export LD_LIBRARY_PATH="/home/steve/.venvs/vllm-xpu-managed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXX="${CXX:-g++}"
export CC="${CC:-gcc}"

exec "$VLLM_BIN" bench latency \
  --model "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --input-len "${INPUT_LEN:-500}" \
  --output-len "${OUTPUT_LEN:-256}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --num-iters "${NUM_ITERS:-1}" \
  --num-iters-warmup "${NUM_ITERS_WARMUP:-1}" \
  --dtype half \
  --quantization inc \
  --language-model-only \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-2G}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  "$@"
