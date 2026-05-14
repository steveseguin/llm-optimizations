#!/usr/bin/env bash
set -euo pipefail

# Quality-cleared four-B70 MiniMax M2.7 AutoRound W4A16 recipe from 2026-05-14.
# Keeps Inductor disabled because compiled decode produced invalid token-0 output.

export TP="${TP:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export INPUT_LEN="${INPUT_LEN:-512}"
export OUTPUT_LEN="${OUTPUT_LEN:-1536}"
export NUM_PROMPTS="${NUM_PROMPTS:-1}"
export DTYPE="${DTYPE:-float16}"
export USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
export XPU_GRAPH="${XPU_GRAPH:-1}"
export VLLM_XPU_FORCE_GRAPH_WITH_COMM="${VLLM_XPU_FORCE_GRAPH_WITH_COMM:-1}"
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE="${VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE:-1}"
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE="${VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE:-1}"
export RUN_TIMEOUT="${RUN_TIMEOUT:-25m}"

export EXTRA_ARGS="${EXTRA_ARGS:---async-engine --block-size 256 --no-enable-prefix-caching --attention-backend TRITON_ATTN --compilation-config {\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_num_of_warmups\":0,\"compile_sizes\":[1]}}"

exec /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
