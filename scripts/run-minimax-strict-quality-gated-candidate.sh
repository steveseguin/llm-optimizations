#!/usr/bin/env bash
set -euo pipefail

# Strict MiniMax M2.7 candidate gate for the current quality-valid 4x B70 path.
# A candidate must pass exact token-hash canaries and semantic canaries on the
# same piecewise graph path as the accepted ~65.75 tok/s baseline before any
# throughput benchmark is run.

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-strict-candidates}"
LABEL="${LABEL:-candidate}"
TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-1536}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
BENCH_REPEATS="${BENCH_REPEATS:-2}"
QUALITY_TIMEOUT="${QUALITY_TIMEOUT:-30m}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-25m}"
RUN_TIMEOUT_KILL_AFTER="${RUN_TIMEOUT_KILL_AFTER:-30s}"
SHM_STALL_MAX_WARNINGS="${SHM_STALL_MAX_WARNINGS:-3}"
CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/fast-ai/vllm-cache-exp/minimax-strict-${LABEL}}"
RAW145_PROMPT="${RAW145_PROMPT:-/home/steve/llm-optimizations-publish/prompts/minimax-raw145-tokenhash-canary.txt}"
RAW145_N64_HASH="${RAW145_N64_HASH:-267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd}"
RAW145_N256_HASH="${RAW145_N256_HASH:-58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537}"

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
stem="minimax-${LABEL}-strict-tp${TP}-ctx${MAX_MODEL_LEN}-mbt${MAX_BATCHED_TOKENS}-bs${BLOCK_SIZE}-${ts}"
summary_json="$OUTDIR/${stem}-summary.json"
quality_dir="$OUTDIR/${stem}-quality"
mkdir -p "$quality_dir"

if ip link show wlxe865d47e3a48 >/dev/null 2>&1; then
  export FI_TCP_IFACE="${FI_TCP_IFACE:-wlxe865d47e3a48}"
  export CCL_KVS_IFACE="${CCL_KVS_IFACE:-wlxe865d47e3a48}"
fi
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0,1,2,3}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,1,2,3}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-1}"
export USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
export VLLM_XPU_USE_LLM_SCALER_MOE="${VLLM_XPU_USE_LLM_SCALER_MOE:-$USE_LLM_SCALER_MOE}"
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_XPU_FORCE_GRAPH_WITH_COMM="${VLLM_XPU_FORCE_GRAPH_WITH_COMM:-1}"
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE="${VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE:-1}"
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE="${VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE:-1}"
export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT="${VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT:-1}"
export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS="${VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS:-2}"
export VLLM_CACHE_ROOT="$CACHE_ROOT"

source "$VENV/bin/activate"

quality_jsons=()
quality_logs=()
quality_names=()

json_array() {
  if [ "$#" -eq 0 ]; then
    printf '[]'
  else
    printf '%s\n' "$@" | jq -R . | jq -s .
  fi
}

run_quality_check() {
  local name="$1"
  shift
  local json="$quality_dir/${name}.json"
  local log="$quality_dir/${name}.log"
  quality_jsons+=("$json")
  quality_logs+=("$log")
  quality_names+=("$name")
  printf 'quality_check=%s\njson=%s\nlog=%s\n' "$name" "$json" "$log"
  timeout --foreground --signal=TERM --kill-after="$RUN_TIMEOUT_KILL_AFTER" "$QUALITY_TIMEOUT" \
    python /home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py \
      --mode graph \
      --model "$MODEL" \
      --out "$json" \
      --raw-prompt \
      --prompt-file "$RAW145_PROMPT" \
      --tensor-parallel-size "$TP" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --block-size "$BLOCK_SIZE" \
      --attention-backend TRITON_ATTN \
      --qk-norm-restore-weight \
      --qk-norm-restore-weight-min-tokens "$VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS" \
      --vllm-cache-root "$CACHE_ROOT" \
      "$@" 2>&1 | tee "$log"
}

run_semantic_suite() {
  local name="semantic-suite-n64-r2"
  local json="$quality_dir/${name}.json"
  local log="$quality_dir/${name}.log"
  quality_jsons+=("$json")
  quality_logs+=("$log")
  quality_names+=("$name")
  printf 'quality_check=%s\njson=%s\nlog=%s\n' "$name" "$json" "$log"
  timeout --foreground --signal=TERM --kill-after="$RUN_TIMEOUT_KILL_AFTER" "$QUALITY_TIMEOUT" \
    python /home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py \
      --mode graph \
      --model "$MODEL" \
      --out "$json" \
      --raw-prompt \
      --max-tokens 64 \
      --runs 2 \
      --prompt-file /home/steve/llm-optimizations-publish/prompts/minimax-pass-canary-raw.txt \
      --prompt-file /home/steve/llm-optimizations-publish/prompts/minimax-arithmetic-canary-raw.txt \
      --prompt-file /home/steve/llm-optimizations-publish/prompts/minimax-code-canary-raw.txt \
      --tensor-parallel-size "$TP" \
      --dtype "$DTYPE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --block-size "$BLOCK_SIZE" \
      --attention-backend TRITON_ATTN \
      --qk-norm-restore-weight \
      --qk-norm-restore-weight-min-tokens "$VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS" \
      --vllm-cache-root "$CACHE_ROOT" \
      --require-substring PASS \
      --require-substring 42 \
      --require-substring "def add_one" \
      --require-regex "return\\s+x\\s*\\+\\s*1" \
      2>&1 | tee "$log"
}

write_summary() {
  local status="$1"
  shift || true
  local bench_jsons=("$@")
  jq -n \
    --arg label "$LABEL" \
    --arg status "$status" \
    --arg model "$MODEL" \
    --arg cache_root "$CACHE_ROOT" \
    --arg timestamp "$ts" \
    --argjson tp "$TP" \
    --argjson max_model_len "$MAX_MODEL_LEN" \
    --argjson max_batched_tokens "$MAX_BATCHED_TOKENS" \
    --argjson max_num_seqs "$MAX_NUM_SEQS" \
    --argjson block_size "$BLOCK_SIZE" \
    --argjson input_len "$INPUT_LEN" \
    --argjson output_len "$OUTPUT_LEN" \
    --argjson bench_repeats "$BENCH_REPEATS" \
    --arg raw145_n64_hash "$RAW145_N64_HASH" \
    --arg raw145_n256_hash "$RAW145_N256_HASH" \
    --argjson quality_names "$(json_array "${quality_names[@]}")" \
    --argjson quality_jsons "$(json_array "${quality_jsons[@]}")" \
    --argjson quality_logs "$(json_array "${quality_logs[@]}")" \
    --argjson bench_jsons "$(json_array "${bench_jsons[@]}")" \
    --arg vllm_minimax_moe_delay_allreduce "${VLLM_MINIMAX_MOE_DELAY_ALLREDUCE:-}" \
    --arg vllm_minimax_dist_residual_allreduce "${VLLM_MINIMAX_M2_DIST_RESIDUAL_ALLREDUCE:-}" \
    --arg vllm_xpu_use_llm_scaler_moe_minimax_logits "${VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS:-}" \
    --arg vllm_xpu_use_llm_scaler_moe_logits "${VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS:-}" \
    --arg vllm_xpu_use_llm_scaler_moe_ws "${VLLM_XPU_USE_LLM_SCALER_MOE_WS:-}" \
    --arg vllm_minimax_m2_candidate_router_topm "${VLLM_MINIMAX_M2_CANDIDATE_ROUTER_TOPM:-}" \
    --arg vllm_minimax_m2_candidate_router_xpu_repair "${VLLM_MINIMAX_M2_CANDIDATE_ROUTER_XPU_REPAIR:-}" \
    --arg vllm_minimax_m2_fp16_router "${VLLM_MINIMAX_M2_FP16_ROUTER:-}" \
    --arg vllm_minimax_qk_rms_xpu_helper "${VLLM_MINIMAX_QK_RMS_XPU_HELPER:-}" \
    --arg vllm_xpu_compile_allreduce_no_clone "${VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE:-}" \
    --arg vllm_xpu_compile_out_of_place_allreduce "${VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE:-}" \
    --arg ccl_topo_fabric_vertex_connection_check "${CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK:-}" \
    --arg ccl_topo_p2p_access "${CCL_TOPO_P2P_ACCESS:-}" \
    '{
      label: $label,
      status: $status,
      timestamp_utc: $timestamp,
      model: $model,
      hardware: "4x Intel Arc Pro B70 32GB",
      runtime: {
        tensor_parallel_size: $tp,
        dtype: "float16",
        max_model_len: $max_model_len,
        max_num_batched_tokens: $max_batched_tokens,
        max_num_seqs: $max_num_seqs,
        block_size: $block_size,
        vllm_cache_root: $cache_root
      },
      quality_policy: {
        raw145_n64_expected_combined_token_sha256: $raw145_n64_hash,
        raw145_n256_expected_combined_token_sha256: $raw145_n256_hash,
        semantic_suite: "PASS, arithmetic 42, add_one function with return x + 1; two greedy repeats must be deterministic"
      },
      candidate_env: {
        VLLM_MINIMAX_MOE_DELAY_ALLREDUCE: $vllm_minimax_moe_delay_allreduce,
        VLLM_MINIMAX_M2_DIST_RESIDUAL_ALLREDUCE: $vllm_minimax_dist_residual_allreduce,
        VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE: env.VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE,
        VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT: env.VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT,
        VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS: env.VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS,
        VLLM_XPU_USE_LLM_SCALER_MOE: env.VLLM_XPU_USE_LLM_SCALER_MOE,
        VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS: $vllm_xpu_use_llm_scaler_moe_minimax_logits,
        VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS: $vllm_xpu_use_llm_scaler_moe_logits,
        VLLM_XPU_USE_LLM_SCALER_MOE_WS: $vllm_xpu_use_llm_scaler_moe_ws,
        VLLM_MINIMAX_M2_CANDIDATE_ROUTER_TOPM: $vllm_minimax_m2_candidate_router_topm,
        VLLM_MINIMAX_M2_CANDIDATE_ROUTER_XPU_REPAIR: $vllm_minimax_m2_candidate_router_xpu_repair,
        VLLM_MINIMAX_M2_FP16_ROUTER: $vllm_minimax_m2_fp16_router,
        VLLM_MINIMAX_QK_RMS_XPU_HELPER: $vllm_minimax_qk_rms_xpu_helper,
        VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE: $vllm_xpu_compile_allreduce_no_clone,
        VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE: $vllm_xpu_compile_out_of_place_allreduce,
        VLLM_XPU_ENABLE_XPU_GRAPH: env.VLLM_XPU_ENABLE_XPU_GRAPH,
        VLLM_XPU_FORCE_GRAPH_WITH_COMM: env.VLLM_XPU_FORCE_GRAPH_WITH_COMM,
        VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE: env.VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE,
        CCL_TOPO_P2P_ACCESS: $ccl_topo_p2p_access,
        CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK: $ccl_topo_fabric_vertex_connection_check
      },
      quality_artifacts: [
        range(0; ($quality_jsons | length)) as $i
        | {name: $quality_names[$i], json: $quality_jsons[$i], log: $quality_logs[$i]}
      ],
      benchmark_policy: {
        prompt_tokens: $input_len,
        output_tokens: $output_len,
        num_prompts: 1,
        repeats_requested: $bench_repeats,
        benchmarked_only_after_quality_pass: true
      },
      benchmark_jsons: $bench_jsons
    }' > "$summary_json"
}

if ! run_quality_check raw145-n64-exact \
  --max-tokens 64 \
  --runs 1 \
  --expected-token-sha256 "$RAW145_N64_HASH"; then
  write_summary quality_failed_raw145_n64
  printf 'summary_json=%s\n' "$summary_json"
  exit 1
fi
if ! run_quality_check raw145-n256-exact \
  --max-tokens 256 \
  --runs 1 \
  --expected-token-sha256 "$RAW145_N256_HASH"; then
  write_summary quality_failed_raw145_n256
  printf 'summary_json=%s\n' "$summary_json"
  exit 1
fi
if ! run_semantic_suite; then
  write_summary quality_failed_semantic_suite
  printf 'summary_json=%s\n' "$summary_json"
  exit 1
fi

bench_jsons=()
bench_logs=()
if [ "$BENCH_REPEATS" -gt 0 ]; then
  for i in $(seq 1 "$BENCH_REPEATS"); do
    run_out="$(
      MODEL="$MODEL" \
      OUTDIR="$OUTDIR" \
      TP="$TP" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS" \
      MAX_NUM_SEQS="$MAX_NUM_SEQS" \
      INPUT_LEN="$INPUT_LEN" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      DTYPE="$DTYPE" \
      USE_LLM_SCALER_MOE="$USE_LLM_SCALER_MOE" \
      XPU_GRAPH=1 \
      RUN_TIMEOUT="$BENCH_TIMEOUT" \
      RUN_TIMEOUT_KILL_AFTER="$RUN_TIMEOUT_KILL_AFTER" \
      SHM_STALL_MAX_WARNINGS="$SHM_STALL_MAX_WARNINGS" \
      EXTRA_ARGS="--async-engine --block-size $BLOCK_SIZE --no-enable-prefix-caching --attention-backend TRITON_ATTN --compilation-config {\"use_inductor_graph_partition\":true,\"compile_sizes\":[1],\"cudagraph_mode\":\"PIECEWISE\"}" \
      /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
    )"
    printf '%s\n' "$run_out"
    bench_jsons+=("$(printf '%s\n' "$run_out" | awk -F= '/^json=/{print $2; exit}')")
    bench_logs+=("$(printf '%s\n' "$run_out" | awk -F= '/^log=/{print $2; exit}')")
  done
fi

write_summary quality_passed "${bench_jsons[@]}"

if [ "${#bench_jsons[@]}" -gt 0 ]; then
  tmp="$(mktemp)"
  for path in "${bench_jsons[@]}"; do
    jq --arg path "$path" --argjson output_tokens "$OUTPUT_LEN" \
      '. + {path: $path, output_tokens_per_second: ($output_tokens / .elapsed_time)}' "$path"
  done | jq -s --slurpfile summary "$summary_json" \
    --argjson logs "$(json_array "${bench_logs[@]}")" '
    $summary[0] + {
      benchmark_logs: $logs,
      benchmarks: .,
      output_toks_per_second: [.[].output_tokens_per_second],
      total_toks_per_second: [.[].tokens_per_second],
      mean_output_toks_per_second: (([.[].output_tokens_per_second] | add) / length),
      mean_total_toks_per_second: (([.[].tokens_per_second] | add) / length)
    }
  ' > "$tmp"
  mv "$tmp" "$summary_json"
fi

printf 'summary_json=%s\n' "$summary_json"
jq -c '{
  label,
  status,
  quality_artifacts,
  mean_output_toks_per_second,
  mean_total_toks_per_second,
  output_toks_per_second,
  total_toks_per_second
}' "$summary_json"
