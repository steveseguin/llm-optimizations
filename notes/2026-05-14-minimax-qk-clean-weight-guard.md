# MiniMax Q/K RMS Clean-Weight Guard

Goal: recover the fast TP4 full-decode XPU graph path without accepting the
token-0/NUL corruption found by the raw prompt quality gates.

## Root Cause

Finite tracing narrowed the failure to MiniMax Q/K RMSNorm weights, not the
checkpoint files. During profile/warmup, layer 58 `q_norm.weight` was finite.
During real prompt execution, the same parameter storage contained NaNs and
large out-of-range values. That produced NaNs in Q after Q/K RMSNorm and then
all token-id `0` output on affected graph runs.

Key trace:

`/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/eager-raw-neutral121-weight-trace-ctx512-20260515T001731Z.trace.jsonl`

The checkpoint safetensors were clean; this looks like runtime parameter
storage corruption or unsafe graph/runtime aliasing around the MiniMax Q/K
RMSNorm path.

## Fix

Added a default-off guard in `MiniMaxText01RMSNormTP`:

- Cache a clean CPU and XPU clone of finite, in-range Q/K RMSNorm weights.
- For graph capture/decode, return the clean XPU clone without CPU waits.
- For prefill or larger token counts, check the live parameter. If it is
  non-finite or wildly out of range, restore from the clean copy.
- Keep checks disabled unless `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`.
- Avoid per-token decode synchronization with
  `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`.

Patch snapshots:

- `patches/vllm-minimax-qk-rms-clean-weight-guard-20260515.patch`
- `patches/minimax-quality-gate-clean-weight-runner-20260515.patch`

## Validation

Strong single-smoke graph canary:

- JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/graph-raw-neutral121-clean-weight-min2-ctx2048-n32-20260515T010220Z.json`
- Result: passed full-decode graph with the raw 121-token prompt that previously
  exposed NUL output.

Quality-gated throughput repeat:

- Quality JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-raw147-repeat-tp4-ctx2048-mbt512-bs256-p512n1536-20260515T011222Z-quality.json`
- Summary JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-raw147-repeat-tp4-ctx2048-mbt512-bs256-p512n1536-20260515T011222Z-summary.json`
- Raw quality prompt token count recorded by tokenizer: `145`
- Quality gate: passed; `0` NUL tokens, `0` non-space control chars, `28`
  distinct generated token ids.
- Throughput repeats: `61.2576` and `61.4404` output tok/s.
- Mean: `61.3490` output tok/s, `81.7987` total tok/s.
- LocalMaxxing submission: `cmp68grii00kdo301g5kqwapp`

This is slightly below the earlier fastest submitted `61.7528` tok/s run, but
it is the strongest quality-preserving graph result so far because it exercises
the raw prompt corruption boundary and keeps the fast graph path enabled.

4096-context follow-up:

- Quality JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-ctx4096-raw145-repeat-tp4-ctx4096-mbt512-bs256-p512n1536-20260515T012656Z-quality.json`
- Summary JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-ctx4096-raw145-repeat-tp4-ctx4096-mbt512-bs256-p512n1536-20260515T012656Z-summary.json`
- Quality gate: passed with the same tokenizer-count `145` raw prompt; `0` NUL
  tokens, `0` non-space control chars, `28` distinct generated token ids.
- Throughput repeats: `60.8727` and `60.9225` output tok/s.
- Mean: `60.8976` output tok/s, `81.1968` total tok/s at
  `max_model_len=4096`.
- LocalMaxxing submission: `cmp68w1mc00kso3016xc7jsgk`

This converts the previously rejected 4096-context full-decode graph path into
a valid result for this prompt/length screen. It still needs broader semantic
canaries before treating 4096 as broadly production-safe.

## Reproduction

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
FI_TCP_IFACE=wlxe865d47e3a48 CCL_KVS_IFACE=wlxe865d47e3a48 \
OUTDIR=/home/steve/bench-results/minimax-m2.7-quality-gated \
LABEL=clean-weight-full-decode-graph-triton-raw147-repeat \
QK_NORM_RESTORE_WEIGHT=1 \
QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2 \
QUALITY_RAW_PROMPT=1 \
QUALITY_PROMPT_FILE=/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/prompts/neutral-raw147-20260514T224104Z.txt \
QUALITY_TOKENS=64 \
QUALITY_RUNS=1 \
QUALITY_REQUIRE_DETERMINISTIC=0 \
QUALITY_MIN_DISTINCT_GENERATED_TOKENS=8 \
QUALITY_MIN_PRINTABLE_NONSPACE_CHARS=20 \
BENCH_REPEATS=2 \
TP=4 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 \
INPUT_LEN=512 OUTPUT_LEN=1536 DTYPE=float16 BLOCK_SIZE=256 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-quality-gated-candidate.sh
```

## Next

- Make a cleaner upstreamable version that initializes the clean weight copy at
  load time instead of first sane forward.
- Expand quality gates from corruption checks to a small semantic canary suite
  before claiming any new speed work.
- Retest 4096 with longer prompts and actual long-context prompts; the current
  repair is proven for the earlier raw-prompt boundary, not for 128k-style use.
- Continue profiling the valid graph path. The likely remaining speed limit is
  still TP communication plus MiniMax MoE dispatch, not model quality.
