#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-quality-gated}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
PROMPT_FILE="${PROMPT_FILE:-/home/steve/llm-optimizations-publish/prompts/minimax-pass-canary-raw.txt}"
REQUIRE_SUBSTRING="${REQUIRE_SUBSTRING:-PASS}"
MODE="${MODE:-eager}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-8}"
TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-off}"
RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUTDIR/minimax-semantic-canary-${MODE}-tp${TP}-${ts}.json"

source "$VENV/bin/activate"

graph_args=()
if [ "$MODE" = "graph" ]; then
  graph_args=(
    --compilation-mode none
    --cudagraph-mode full_decode_only
    --cudagraph-num-warmups 0
  )
else
  graph_args=(
    --compilation-mode none
    --cudagraph-mode none
    --cudagraph-num-warmups 0
  )
fi

timeout --foreground --signal=TERM --kill-after=30s "$RUN_TIMEOUT" \
  python /home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py \
    --mode "$MODE" \
    --raw-prompt \
    --model "$MODEL" \
    --out "$out" \
    --max-tokens "$MAX_TOKENS" \
    --runs "$RUNS" \
    --prompt-file "$PROMPT_FILE" \
    --tensor-parallel-size "$TP" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size "$BLOCK_SIZE" \
    --attention-backend TRITON_ATTN \
    --async-scheduling "$ASYNC_SCHEDULING" \
    --require-substring "$REQUIRE_SUBSTRING" \
    "${graph_args[@]}"

printf 'json=%s\n' "$out"
jq -c '{
  passed,
  failure_reasons,
  deterministic_across_runs,
  combined_token_sha256,
  output_texts: [.run_records[].prompts[0].text],
  quality_checks: {
    distinct_generated_token_count: .quality_checks.distinct_generated_token_count,
    printable_nonspace_text_chars: .quality_checks.printable_nonspace_text_chars,
    control_nonspace_text_chars: .quality_checks.control_nonspace_text_chars,
    nul_token_count: .quality_checks.nul_token_count,
    degenerate_output: .quality_checks.degenerate_output,
    semantic_checks: .quality_checks.semantic_checks
  }
}' "$out"
