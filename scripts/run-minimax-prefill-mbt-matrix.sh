#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:-/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround}"
CACHE_BASE="${CACHE_BASE:-/mnt/fast-ai/vllm-cache-exp}"
OUTDIR="${OUTDIR:-/home/steve/bench-results/minimax-m2.7-autoround-vllm}"
SUMMARY_DIR="${SUMMARY_DIR:-/home/steve/bench-results/minimax-m2.7-prefill-matrix}"
MATRIX_TS="${MATRIX_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

MBT_VALUES="${MBT_VALUES:-512 1024 2048 4096}"
COLD_RUNS="${COLD_RUNS:-1}"
WARM_RUNS="${WARM_RUNS:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-20m}"

TP="${TP:-4}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
INPUT_LEN="${INPUT_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"

mkdir -p "$SUMMARY_DIR"
run_manifest="$SUMMARY_DIR/minimax-prefill-mbt-matrix-$MATRIX_TS.runs.jsonl"
: > "$run_manifest"

bench_once() {
  local mbt="$1"
  local phase="$2"
  local run_index="$3"
  local cache_root="$CACHE_BASE/minimax-prefill-mbt${mbt}-p${INPUT_LEN}n${OUTPUT_LEN}-${MATRIX_TS}"
  local run_out="$SUMMARY_DIR/minimax-prefill-mbt${mbt}-${phase}-${run_index}-${MATRIX_TS}.out"

  echo "mbt=$mbt phase=$phase run=$run_index cache_root=$cache_root"
  (
    cd "$ROOT"
    export MODEL VLLM_CACHE_ROOT="$cache_root" OUTDIR TP DTYPE MAX_MODEL_LEN
    export MAX_BATCHED_TOKENS="$mbt" MAX_NUM_SEQS INPUT_LEN OUTPUT_LEN
    export NUM_PROMPTS RUN_TIMEOUT
    export USE_LLM_SCALER_MOE=1
    export VLLM_XPU_USE_LLM_SCALER_MOE=1
    export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
    export FORCE_WARMUP=0
    export WARMUP_IF_MISSING=0
    export REQUIRE_WARMUP_SUCCESS=0
    export EXTRA_ARGS="--async-engine --block-size $BLOCK_SIZE --no-enable-prefix-caching --compilation-config {\"use_inductor_graph_partition\":true,\"compile_sizes\":[1],\"cudagraph_mode\":\"PIECEWISE\"}"
    "$SCRIPT_DIR/bench-vllm-minimax-autoround-xpu-warm-aot.sh"
  ) | tee "$run_out"

  local json_path
  json_path="$(sed -n 's/^json=//p' "$run_out" | tail -1)"
  local log_path
  log_path="$(sed -n 's/^log=//p' "$run_out" | tail -1)"
  python3 - "$run_manifest" "$mbt" "$phase" "$run_index" "$json_path" "$log_path" "$cache_root" <<'PY'
import json
import sys
from pathlib import Path

manifest, mbt, phase, run_index, json_path, log_path, cache_root = sys.argv[1:]
record = {
    "max_batched_tokens": int(mbt),
    "phase": phase,
    "run_index": int(run_index),
    "json": json_path,
    "log": log_path,
    "cache_root": cache_root,
}
path = Path(json_path)
if path.exists() and path.stat().st_size:
    data = json.loads(path.read_text())
    elapsed = float(data["elapsed_time"])
    record.update(
        {
            "success": True,
            "elapsed_s": elapsed,
            "tok_s_out": 512 / elapsed,
            "tok_s_total_reported": float(data["tokens_per_second"]),
            "total_num_tokens_reported": data.get("total_num_tokens"),
        }
    )
else:
    record["success"] = False
with Path(manifest).open("a") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
}

for mbt in $MBT_VALUES; do
  for run in $(seq 1 "$COLD_RUNS"); do
    bench_once "$mbt" cold "$run"
  done
  for run in $(seq 1 "$WARM_RUNS"); do
    bench_once "$mbt" warm "$run"
  done
done

summary="$SUMMARY_DIR/minimax-prefill-mbt-matrix-$MATRIX_TS.summary.json"
python3 - "$run_manifest" "$summary" "$INPUT_LEN" "$OUTPUT_LEN" "$NUM_PROMPTS" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

manifest, summary_path, input_len, output_len, num_prompts = sys.argv[1:]
runs = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
by_mbt: dict[int, list[dict]] = defaultdict(list)
for run in runs:
    if run.get("success") and run["phase"] == "warm":
        by_mbt[run["max_batched_tokens"]].append(run)

summary = {
    "shape": {
        "input_len": int(input_len),
        "output_len": int(output_len),
        "num_prompts": int(num_prompts),
    },
    "runs": runs,
    "warm_summary_by_mbt": {},
}
for mbt, items in sorted(by_mbt.items()):
    outs = [item["tok_s_out"] for item in items]
    totals = [item["tok_s_total_reported"] for item in items]
    summary["warm_summary_by_mbt"][str(mbt)] = {
        "runs": len(items),
        "tok_s_out_mean": statistics.mean(outs),
        "tok_s_out_min": min(outs),
        "tok_s_out_max": max(outs),
        "tok_s_total_mean": statistics.mean(totals),
        "tok_s_total_min": min(totals),
        "tok_s_total_max": max(totals),
    }

Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["warm_summary_by_mbt"], indent=2, sort_keys=True))
PY
