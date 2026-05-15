#!/usr/bin/env bash
set -euo pipefail

# Run a MiniMax M2.7 AutoRound candidate only after the matching runtime passes
# the deterministic chat-template quality smoke. This keeps throughput wins from
# being promoted if graph/compiler changes corrupt generated tokens.

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-quality-gated}"
TP="${TP:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-1536}"
DTYPE="${DTYPE:-float16}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
QUALITY_TOKENS="${QUALITY_TOKENS:-160}"
QUALITY_RUNS="${QUALITY_RUNS:-1}"
QUALITY_PROMPT_FILE="${QUALITY_PROMPT_FILE:-/home/steve/llm-optimizations-publish/prompts/minimax-long-context-quality-smoke.txt}"
QUALITY_RAW_PROMPT="${QUALITY_RAW_PROMPT:-0}"
QUALITY_EXPECTED_TOKEN_SHA256="${QUALITY_EXPECTED_TOKEN_SHA256:-}"
QUALITY_REQUIRE_DETERMINISTIC="${QUALITY_REQUIRE_DETERMINISTIC:-0}"
QUALITY_ASYNC_SCHEDULING="${QUALITY_ASYNC_SCHEDULING:-default}"
QUALITY_MIN_DISTINCT_GENERATED_TOKENS="${QUALITY_MIN_DISTINCT_GENERATED_TOKENS:-2}"
QUALITY_MIN_PRINTABLE_NONSPACE_CHARS="${QUALITY_MIN_PRINTABLE_NONSPACE_CHARS:-1}"
QUALITY_MAX_CONTROL_NONSPACE_CHARS="${QUALITY_MAX_CONTROL_NONSPACE_CHARS:-0}"
QUALITY_MAX_NUL_TOKEN_COUNT="${QUALITY_MAX_NUL_TOKEN_COUNT:-0}"
QUALITY_REQUIRE_SUBSTRING="${QUALITY_REQUIRE_SUBSTRING:-}"
QUALITY_REQUIRE_REGEX="${QUALITY_REQUIRE_REGEX:-}"
QUALITY_TIMEOUT="${QUALITY_TIMEOUT:-25m}"
BENCH_REPEATS="${BENCH_REPEATS:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-25m}"
SHM_STALL_MAX_WARNINGS="${SHM_STALL_MAX_WARNINGS:-0}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
LABEL="${LABEL:-full-decode-graph-triton}"
QK_NORM_RESTORE_WEIGHT="${QK_NORM_RESTORE_WEIGHT:-0}"
QK_NORM_RESTORE_WEIGHT_MIN_TOKENS="${QK_NORM_RESTORE_WEIGHT_MIN_TOKENS:-2}"

export USE_LLM_SCALER_MOE="${USE_LLM_SCALER_MOE:-1}"
export XPU_GRAPH="${XPU_GRAPH:-1}"
export VLLM_XPU_FORCE_GRAPH_WITH_COMM="${VLLM_XPU_FORCE_GRAPH_WITH_COMM:-1}"
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE="${VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE:-1}"
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE="${VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE:-1}"
if [ "$QK_NORM_RESTORE_WEIGHT" = "1" ]; then
  export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
  export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS="$QK_NORM_RESTORE_WEIGHT_MIN_TOKENS"
else
  unset VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT
  unset VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS
fi

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
stem="minimax-${LABEL}-tp${TP}-ctx${MAX_MODEL_LEN}-mbt${MAX_BATCHED_TOKENS}-bs${BLOCK_SIZE}-p${INPUT_LEN}n${OUTPUT_LEN}-${ts}"
quality_json="$OUTDIR/${stem}-quality.json"
quality_log="$OUTDIR/${stem}-quality.log"
summary_json="$OUTDIR/${stem}-summary.json"

source "$VENV/bin/activate"

quality_cmd=(
  python /home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py
  --mode graph
  --model "$MODEL"
  --out "$quality_json"
  --max-tokens "$QUALITY_TOKENS"
  --runs "$QUALITY_RUNS"
  --prompt-file "$QUALITY_PROMPT_FILE"
  --tensor-parallel-size "$TP"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --block-size "$BLOCK_SIZE"
  --compilation-mode none
  --cudagraph-mode full_decode_only
  --cudagraph-num-warmups 0
  --attention-backend TRITON_ATTN
  --async-scheduling "$QUALITY_ASYNC_SCHEDULING"
  --min-distinct-generated-tokens "$QUALITY_MIN_DISTINCT_GENERATED_TOKENS"
  --min-printable-nonspace-chars "$QUALITY_MIN_PRINTABLE_NONSPACE_CHARS"
  --max-control-nonspace-chars "$QUALITY_MAX_CONTROL_NONSPACE_CHARS"
  --max-nul-token-count "$QUALITY_MAX_NUL_TOKEN_COUNT"
)

if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  quality_cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [ "$QUALITY_RAW_PROMPT" = "1" ]; then
  quality_cmd+=(--raw-prompt)
fi
if [ "$QK_NORM_RESTORE_WEIGHT" = "1" ]; then
  quality_cmd+=(
    --qk-norm-restore-weight
    --qk-norm-restore-weight-min-tokens "$QK_NORM_RESTORE_WEIGHT_MIN_TOKENS"
  )
fi
if [ -n "$QUALITY_EXPECTED_TOKEN_SHA256" ]; then
  quality_cmd+=(--expected-token-sha256 "$QUALITY_EXPECTED_TOKEN_SHA256")
fi
if [ -n "$QUALITY_REQUIRE_SUBSTRING" ]; then
  quality_cmd+=(--require-substring "$QUALITY_REQUIRE_SUBSTRING")
fi
if [ -n "$QUALITY_REQUIRE_REGEX" ]; then
  quality_cmd+=(--require-regex "$QUALITY_REQUIRE_REGEX")
fi
if [ "$QUALITY_REQUIRE_DETERMINISTIC" != "1" ]; then
  quality_cmd+=(--allow-nondeterministic-output)
fi

printf 'quality_json=%s\n' "$quality_json"
printf 'quality_log=%s\n' "$quality_log"
timeout --foreground --signal=TERM --kill-after=30s "$QUALITY_TIMEOUT" \
  "${quality_cmd[@]}" 2>&1 | tee "$quality_log"

bench_jsons=()
bench_logs=()
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
    DTYPE="$DTYPE" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    RUN_TIMEOUT="$RUN_TIMEOUT" \
    SHM_STALL_MAX_WARNINGS="$SHM_STALL_MAX_WARNINGS" \
    EXTRA_ARGS="--async-engine --block-size $BLOCK_SIZE --no-enable-prefix-caching --attention-backend TRITON_ATTN --compilation-config {\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_num_of_warmups\":0,\"compile_sizes\":[1]}" \
    /home/steve/llm-optimizations-publish/scripts/run-minimax-full-decode-graph-triton.sh
  )"
  printf '%s\n' "$run_out"
  bench_jsons+=("$(printf '%s\n' "$run_out" | awk -F= '/^json=/{print $2; exit}')")
  bench_logs+=("$(printf '%s\n' "$run_out" | awk -F= '/^log=/{print $2; exit}')")
done

jq -n \
  --arg label "$LABEL" \
  --arg quality_json "$quality_json" \
  --arg quality_log "$quality_log" \
  --argjson prompt_tokens "$INPUT_LEN" \
  --argjson output_tokens "$OUTPUT_LEN" \
  --argjson context_length "$MAX_MODEL_LEN" \
  --argjson max_num_batched_tokens "$MAX_BATCHED_TOKENS" \
  --argjson block_size "$BLOCK_SIZE" \
  --arg gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --argjson qk_norm_restore_weight "$QK_NORM_RESTORE_WEIGHT" \
  --argjson qk_norm_restore_weight_min_tokens "$QK_NORM_RESTORE_WEIGHT_MIN_TOKENS" \
  --argjson bench_jsons "$(printf '%s\n' "${bench_jsons[@]}" | jq -R . | jq -s .)" \
  --argjson bench_logs "$(printf '%s\n' "${bench_logs[@]}" | jq -R . | jq -s .)" \
  '{
    label: $label,
    quality_json: $quality_json,
    quality_log: $quality_log,
    prompt_tokens: $prompt_tokens,
    output_tokens: $output_tokens,
    context_length: $context_length,
    max_num_batched_tokens: $max_num_batched_tokens,
    block_size: $block_size,
    gpu_memory_utilization: (if $gpu_memory_utilization == "" then null else ($gpu_memory_utilization | tonumber) end),
    qk_norm_restore_weight: ($qk_norm_restore_weight == 1),
    qk_norm_restore_weight_min_tokens: $qk_norm_restore_weight_min_tokens,
    bench_jsons: $bench_jsons,
    bench_logs: $bench_logs
  }' > "$summary_json"

tmp="$(mktemp)"
for path in "${bench_jsons[@]}"; do
  jq --arg path "$path" --argjson output_tokens "$OUTPUT_LEN" \
    '. + {path: $path, output_tokens_per_second: ($output_tokens / .elapsed_time)}' "$path"
done | jq -s --slurpfile summary "$summary_json" '
  $summary[0] + {
    benchmarks: .,
    output_toks_per_second: [.[].output_tokens_per_second],
    total_toks_per_second: [.[].tokens_per_second],
    mean_output_toks_per_second: (([.[].output_tokens_per_second] | add) / length),
    mean_total_toks_per_second: (([.[].tokens_per_second] | add) / length)
  }' > "$tmp"
mv "$tmp" "$summary_json"

printf 'summary_json=%s\n' "$summary_json"
jq -c '{
  label,
  quality_json,
  mean_output_toks_per_second,
  mean_total_toks_per_second,
  output_toks_per_second,
  total_toks_per_second,
  benchmarks: [.benchmarks[] | {path, elapsed_time, tokens_per_second, output_tokens_per_second}]
}' "$summary_json"
