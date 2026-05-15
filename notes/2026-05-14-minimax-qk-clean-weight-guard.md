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
- Seed the clean copy in the weight loader and attach it to the parameter, so
  graph capture does not depend on a prior sane forward to initialize the
  fallback copy.
- For graph capture/decode, return the clean XPU clone without CPU waits.
- For prefill or larger token counts, check the live parameter. If it is
  non-finite or wildly out of range, restore from the clean copy.
- Keep checks disabled unless `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`.
- Avoid per-token decode synchronization with
  `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`.

Patch snapshots:

- `patches/vllm-minimax-qk-rms-clean-weight-guard-20260515.patch`
- `patches/minimax-quality-gate-clean-weight-runner-20260515.patch`

Loader-seeded refinement validation:

- JSON: `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/graph-raw121-loader-clean-weight-ctx2048-n32-20260515T013655Z.json`
- Result: passed full-decode graph canary after moving clean-copy seeding into
  `weight_loader`; `0` NUL tokens, `0` non-space control chars, `27` distinct
  generated token ids.
- Local record:
  `data/minimax-m27-loader-seeded-clean-weight-canary-20260515.json`

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

Server latency probe:

- Result JSON:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics/vllm-minimax-m27-autoround-serve-tp4-p512n256-np3-20260515T014336Z.json`
- Local record:
  `data/minimax-m27-serve-latency-ctx4096-clean-weight-20260515.json`
- Config: `max_model_len=4096`, `3` random prompts, `512` input tokens,
  `256` output tokens, `temperature=0`, `MAX_NUM_SEQS=1`.
- Output throughput: `60.7175` tok/s.
- Total token throughput: `182.1524` tok/s for the serving benchmark window.
- Mean TTFT: `4894.95` ms; median TTFT: `4901.29` ms.
- Mean TPOT: `14.6272` ms; median TPOT: `14.6256` ms.
- Mean ITL: `14.6272` ms; median ITL: `14.6157` ms.

This is not a LocalMaxxing submission because it is a server-latency probe with
multiple queued requests and shorter output length. Use it to track interactive
latency and prefill/first-token behavior, not as the headline decode result.

Piecewise/AOT compiled path repair:

- Short canary JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/compiled-piecewise-raw145-clean-weight-dynamo-safe-ctx2048-n64-20260515T015044Z.json`
- Longer canary JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/compiled-piecewise-raw145-clean-weight-dynamo-safe-ctx2048-n256-20260515T020231Z.json`
- Throughput summary:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/piecewise-throughput/minimax-clean-weight-piecewise-tp4-ctx2048-mbt512-bs256-p512n1536-3run-20260515T020502Z-summary.json`
- Local records:
  - `data/minimax-m27-clean-weight-piecewise-aot-quality64-20260515.json`
  - `data/minimax-m27-clean-weight-piecewise-aot-quality256-20260515.json`
  - `data/minimax-m27-clean-weight-piecewise-aot-p512n1536-3run-20260515.json`
- Quality gates: both raw-prompt canaries passed on the compiled piecewise/AOT
  recipe; the 256-token canary generated `0` NUL tokens, `0` non-space control
  chars, and `28` distinct generated token ids.
- Throughput repeats: `64.6223`, `66.6589`, and `65.9762` output tok/s.
- Mean: `65.7525` output tok/s, `87.6699` total tok/s.
- LocalMaxxing submission: `cmp6a5c1o00mpo3011hg8ncyp`

This is the first repaired piecewise/AOT result promoted after the raw-prompt
NUL-token failure was understood. It is slower than the old invalid `~73` tok/s
speed-only diagnostic, but faster than the earlier quality-corrected
full-decode graph baseline. The important distinction is that this result keeps
the same model/quantization and passes the corruption gates.

Follow-up controls:

- Local record: `data/minimax-m27-clean-weight-followups-20260515.json`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=100000` passed the 64-token
  raw quality gate but reached only `64.8187` output tok/s on one p512/n1536
  sample, below the promoted `65.7525` mean. It is not a speed route.
- The same min-token run exposed a driver/compiler issue during graph capture:
  `ocloc`/IGC returned an internal compiler error with a floating-point
  exception for a generated Triton reduction kernel, then vLLM recovered and
  produced clean output. Keep this as a B70 driver/compiler stability clue.
- Patched the optional MiniMax Q/K RMS XPU helper so it uses clean guarded
  Q/K norm weights when `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`. The helper-enabled
  raw quality gate passed, but one p512/n1536 sample reached only `64.9878`
  output tok/s, so the helper remains off for the headline recipe.
- Forcing `CCL_ZE_IPC_EXCHANGE=pidfd` through `CCL_IPC=pidfd` was also a
  negative on the repaired piecewise/AOT path: one p512/n1536 sample reached
  only `48.9748` output tok/s and `65.2997` total tok/s. Leave IPC exchange at
  the default for the current headline recipe.
- Patch snapshot:
  `patches/vllm-minimax-qk-rms-helper-clean-weight-guard-20260515.patch`

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

- Convert the guard into a cleaner upstreamable version, likely with a named
  corruption workaround flag and less diagnostic tracing.
- Expand quality gates from corruption checks to a small semantic canary suite
  before claiming any new speed work.
- Retest 4096 with longer prompts and actual long-context prompts; the current
  repair is proven for the earlier raw-prompt boundary, not for 128k-style use.
- Continue profiling the valid graph path. The likely remaining speed limit is
  still TP communication plus MiniMax MoE dispatch, not model quality.
- For the compiled recipe, the clean-weight guard needed a Dynamo-safe branch:
  under `torch.compiler.is_compiling()`, return the clean XPU copy directly and
  avoid calling stream-capture inspection from inside Dynamo tracing.
