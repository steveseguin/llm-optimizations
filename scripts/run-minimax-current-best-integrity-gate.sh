#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
VENV="${VENV:-/home/steve/.venvs/vllm-xpu}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-20260513T171301Z}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-integrity-gate}"
REPEAT_RUNS="${REPEAT_RUNS:-3}"
RUN_QUALITY="${RUN_QUALITY:-1}"
RUN_REPEAT="${RUN_REPEAT:-1}"
RUN_PREFILL="${RUN_PREFILL:-0}"

TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-1536}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
TARGET_OUTPUT_TOK_S="${TARGET_OUTPUT_TOK_S:-73.30631164343902}"
MIN_RATIO="${MIN_RATIO:-0.985}"
MAX_CV_PCT="${MAX_CV_PCT:-1.0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-15m}"

mkdir -p "$OUTDIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
gate_dir="$OUTDIR/current-best-gate-$ts"
mkdir -p "$gate_dir"

aot_cache="${AOT_CACHE:-}"
if [ -z "$aot_cache" ]; then
  rank_dir="$(
    find "$VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile" \
      -path '*/rank_0_0/model' -type f -printf '%h\n' 2>/dev/null \
      | sort | tail -1
  )"
  if [ -n "$rank_dir" ]; then
    aot_cache="$(dirname "$rank_dir")/inductor_cache"
  fi
fi
if [ -z "$aot_cache" ]; then
  echo "No AOT cache found under $VLLM_CACHE_ROOT" >&2
  exit 1
fi

echo "gate_dir=$gate_dir"
echo "aot_cache=$aot_cache"
"$SCRIPT_DIR/validate-minimax-aot-collectives.py" \
  "$aot_cache" \
  --out "$gate_dir/aot-collectives.json" \
  --strict

if [ "$RUN_QUALITY" = "1" ]; then
  LLM_SCALER_KERNELS="${LLM_SCALER_KERNELS:-/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python}"
  export PYTHONPATH="$LLM_SCALER_KERNELS:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$VENV/lib:$VENV/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
  source "$VENV/bin/activate"
  "$SCRIPT_DIR/run-vllm-minimax-quality-check.py" \
    --mode graph \
    --model "$MODEL" \
    --out "$gate_dir/quality-smoke.json" \
    --max-tokens "${QUALITY_MAX_TOKENS:-96}" \
    --runs "${QUALITY_RUNS:-2}" \
    --tensor-parallel-size "$TP" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size 256 \
    --disable-prefix-caching \
    --vllm-cache-root "$VLLM_CACHE_ROOT"
fi

repeat_jsons=()
if [ "$RUN_REPEAT" = "1" ]; then
  for i in $(seq 1 "$REPEAT_RUNS"); do
    echo "repeat_run=$i/$REPEAT_RUNS"
    run_out="$gate_dir/repeat-$i.out"
    (
      cd "$ROOT"
      export MODEL VLLM_CACHE_ROOT TP DTYPE MAX_MODEL_LEN MAX_BATCHED_TOKENS
      export MAX_NUM_SEQS INPUT_LEN OUTPUT_LEN NUM_PROMPTS RUN_TIMEOUT
      export USE_LLM_SCALER_MOE=1
      export VLLM_XPU_USE_LLM_SCALER_MOE=1
      export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
      export FORCE_WARMUP=0
      export WARMUP_IF_MISSING=0
      export REQUIRE_WARMUP_SUCCESS=0
      export OUTDIR=/home/steve/bench-results/minimax-m2.7-autoround-vllm
      export EXTRA_ARGS='--async-engine --block-size 256 --no-enable-prefix-caching --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
      "$SCRIPT_DIR/bench-vllm-minimax-autoround-xpu-warm-aot.sh"
    ) | tee "$run_out"
    json_path="$(sed -n 's/^json=//p' "$run_out" | tail -1)"
    if [ -z "$json_path" ] || [ ! -s "$json_path" ]; then
      echo "Missing benchmark JSON for repeat $i" >&2
      exit 1
    fi
    repeat_jsons+=("$json_path")
  done
  "$SCRIPT_DIR/summarize-vllm-repeatability.py" \
    --out "$gate_dir/repeatability-summary.json" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --target-output-tok-s "$TARGET_OUTPUT_TOK_S" \
    --min-ratio "$MIN_RATIO" \
    --max-cv-pct "$MAX_CV_PCT" \
    "${repeat_jsons[@]}"
fi

if [ "$RUN_PREFILL" = "1" ]; then
  echo "prefill_screen=enabled"
  (
    cd "$ROOT"
    export MODEL VLLM_CACHE_ROOT TP DTYPE MAX_MODEL_LEN="${PREFILL_MAX_MODEL_LEN:-8192}"
    export MAX_BATCHED_TOKENS="${PREFILL_MAX_BATCHED_TOKENS:-4096}"
    export MAX_NUM_SEQS=1 INPUT_LEN="${PREFILL_INPUT_LEN:-4096}"
    export OUTPUT_LEN="${PREFILL_OUTPUT_LEN:-512}" NUM_PROMPTS=1 RUN_TIMEOUT
    export USE_LLM_SCALER_MOE=1
    export VLLM_XPU_USE_LLM_SCALER_MOE=1
    export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
    export FORCE_WARMUP=0 WARMUP_IF_MISSING=0 REQUIRE_WARMUP_SUCCESS=0
    export OUTDIR=/home/steve/bench-results/minimax-m2.7-autoround-vllm
    export EXTRA_ARGS='--async-engine --block-size 256 --no-enable-prefix-caching --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
    "$SCRIPT_DIR/bench-vllm-minimax-autoround-xpu-warm-aot.sh"
  ) | tee "$gate_dir/prefill-screen.out"
fi

summary="$gate_dir/gate-summary.json"
python3 - "$gate_dir" "$summary" <<'PY'
import json
import sys
from pathlib import Path

gate_dir = Path(sys.argv[1])
summary = Path(sys.argv[2])
record = {
    "gate_dir": str(gate_dir),
}
aot = json.loads((gate_dir / "aot-collectives.json").read_text())
classification = aot["classification"]
record["aot_collectives"] = {
    "passed": aot["passed"],
    "failures": aot["failures"],
    "actual_allreduce_call_lines": classification["actual_allreduce_call_lines"],
    "actual_wait_tensor_call_lines": classification["actual_wait_tensor_call_lines"],
    "actual_allreduce_wait_pairs_within_7_lines": classification[
        "actual_allreduce_wait_pairs_within_7_lines"
    ],
    "actual_allreduce_categories": classification["actual_allreduce_categories"],
    "wait_gap_lines": classification["wait_gap_lines"],
    "full_record": str(gate_dir / "aot-collectives.json"),
}
quality = gate_dir / "quality-smoke.json"
if quality.exists():
    q = json.loads(quality.read_text())
    first_prompt = {}
    if q.get("run_records"):
        prompts = q["run_records"][0].get("prompts", [])
        if prompts:
            first = prompts[0]
            first_prompt = {
                "text_prefix": first.get("text", "")[:160],
                "token_ids_prefix": first.get("token_ids", [])[:32],
            }
    record["quality_smoke"] = {
        "deterministic_across_runs": q["deterministic_across_runs"],
        "combined_token_sha256": q["combined_token_sha256"],
        "combined_text_sha256": q["combined_text_sha256"],
        "quality_checks": q.get("quality_checks"),
        "first_prompt": first_prompt,
        "runs": q["runs"],
        "n_prompts": q["n_prompts"],
    }
repeatability = gate_dir / "repeatability-summary.json"
if repeatability.exists():
    record["repeatability"] = json.loads(repeatability.read_text())
summary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY
