# MiniMax Quality Gate And Reliability Follow-up

This pass tightened the MiniMax M2.7 AutoRound validation harness after the
`61.75` output tok/s full-decode graph result. The goal is to keep speed work
behind an explicit quality/reliability gate rather than promoting throughput
alone.

## Harness Changes

- Added `--prompt-file` to `scripts/run-vllm-minimax-quality-check.py`.
- Added `--gpu-memory-utilization` to the same script so larger-context
  candidates can be quality-checked with the same memory setting used by the
  benchmark.
- Changed quality prompt execution to run prompts serially instead of batching
  them together. This avoids accidental multi-sequence behavior when
  `max_num_seqs=1`.
- Added `--allow-nondeterministic-output`. The script still records token
  hashes and determinism, but can be configured to fail only on corruption
  (`NUL`, control output, degenerate output) while recording token-hash drift as
  a reliability caveat.
- Added `scripts/run-minimax-quality-gated-candidate.sh`, which runs a
  long-context quality smoke before throughput repeats.
- Fixed `scripts/run-minimax-full-decode-graph-triton.sh` so overriding
  `EXTRA_ARGS` no longer appends an extra `}` to JSON compilation config.

## Fresh Quality Smoke

The checked-in long-context prompt passed under the current full-decode graph
recipe:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-baseline-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T102248Z-quality.json`
- `combined_token_sha256=8d93f5392db36783e4eb3119335a9b6fc0a0470da3c8a3af8c61e8e39efd707e`
- `nul_token_count=0`
- `control_char_output=false`
- `degenerate_output=false`
- `distinct_generated_token_count=88`

This matches the original known-good one-shot long-context quality hash from
the first full-decode graph promotion.

## Reliability Caveats Found

- A first multi-prompt/two-run gate attempt stalled with repeated
  `No available shared memory broadcast block found in 60 seconds`. It was
  interrupted and no throughput result was accepted.
- Two-run greedy generation under the graph path produced coherent text without
  NUL/control output, but token hashes differed between runs in one probe. That
  means exact token-level determinism is not currently proven for this
  full-decode graph path.
- Re-running the same two-run greedy probe with graph disabled and asynchronous
  scheduling disabled still produced different token hashes while generating
  coherent text. This points away from XPU graph capture and the async scheduler
  as the only cause of free-form token drift.
- Disabling the llm-scaler MiniMax INT4 MoE path for a short eager TP4 control
  probe did not restore exact token determinism. The remaining likely causes are
  XPU/oneCCL reduction order, low-margin next-token choices, or other numerical
  nondeterminism below the graph/compiler layer.
- A later two-run quality attempt hung after model load, before graph capture.
  Treat this as a harness/runtime reliability issue to debug before requiring
  multi-run token-hash determinism.
- After fixing the wrapper JSON bug, a direct throughput rerun hung during
  distributed initialization after several interrupted experiments. It was
  killed and is not a benchmark datapoint.
- A short graph-mode semantic canary also stalled with the shared-memory
  broadcast warning after model load. The same final-answer canary succeeds in
  eager mode, so this is being tracked as a graph runtime/harness reliability
  issue rather than a model quality failure.

## Semantic Canary Added

MiniMax's chat template starts generation inside `<think>`, so a one-word chat
prompt measures reasoning text instead of final-answer compliance. To get a
useful canary, this pass added a raw prompt that pre-closes `<think>` and asks
for the final answer token:

- Prompt: `prompts/minimax-pass-canary-raw.txt`
- Runner: `scripts/run-minimax-semantic-canary.sh`
- Eager TP4 result:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/eager-pass-raw-canary-r3-20260514T105708Z.json`
- Three repeated runs all generated `PASS`.
- `deterministic_across_runs=true`
- `nul_token_count=0`
- `control_char_output=false`
- `degenerate_output=false`

The quality checker now records a `passed` boolean, explicit
`failure_reasons`, per-prompt corruption stats, optional required substrings,
optional required regexes, and the async-scheduling override used for a run.

## Current Policy

- The `61.75` output tok/s MiniMax result remains the current fastest
  quality-cleared speed datapoint because it has a clean long-context
  corruption smoke and tight throughput repeats.
- Do not claim token-exact deterministic decoding for this graph recipe yet.
- Future speed candidates must at minimum pass the one-shot long-context
  corruption gate, pass the raw semantic canary, and produce repeatable
  throughput. Stronger promotion should also resolve the two-run hash
  instability/hang or explain why exact token determinism is not expected on XPU
  graph.

## Fresh Repeatability Check

After the harness changes, the current full-decode graph recipe reproduced above
60 output tok/s:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T110046Z-summary.json`
- Graph-path long-context quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T110046Z-quality.json`
- Quality gate: `passed=true`, `nul_token_count=0`,
  `control_char_output=false`, `degenerate_output=false`
- Throughput repeats: `61.3575` and `60.8042` output tok/s
- Mean output throughput: `61.0808` tok/s
- Mean total throughput: `81.4411` tok/s
- LocalMaxxing submission: `cmp5e0t6w007ho301nw1qq45h`

The wrapper summary step initially failed because this host's `jq` rejects the
legacy `--argfile` option. The script now uses `--slurpfile`, and the summary
above was reconstructed from the completed benchmark JSON files.

## Expanded Raw Semantic Canary Suite

Added:

- `prompts/minimax-arithmetic-canary-raw.txt`
- `prompts/minimax-code-canary-raw.txt`
- `scripts/run-minimax-semantic-canary-suite.sh`

The suite now checks three raw, pre-closed-`<think>` prompts:

- final-answer compliance: must include `PASS`
- arithmetic: must include `42`
- short Python code: must include `def add_one` and match
  `return\s+x\s*\+\s*1`

Eager TP4 result:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-semantic-canary-suite-eager-tp4-20260514T113538Z.json`
- `passed=true`
- `deterministic_across_runs=true`
- `combined_token_sha256=adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- `nul_token_count=0`
- `control_nonspace_text_chars=0`
- `degenerate_output=false`
- Run 1 and run 2 outputs matched exactly:
  - `PASS`
  - `42`
  - `def add_one(x): return x + 1`

This does not prove broad benchmark quality, but it raises the bar above a
single PASS token and gives speed experiments a cheap correctness screen before
throughput numbers are promoted.

## Post-reset Graph Semantic Canary

After EP Attempt 13 forced xe devcoredumps and `xpu-smi` device resets, the
promoted TP4 full-decode graph recipe was rechecked with the expanded raw
semantic suite:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-post-reset-full-decode-graph-semantic-canary-20260514T155344Z.json`
- Runtime: TP4, TRITON_ATTN, `CompilationMode.NONE`,
  `FULL_DECODE_ONLY` graph, `block-size=256`,
  llm-scaler INT4 MoE decode enabled
- `passed=true`
- `nul_token_count=0`
- `control_nonspace_text_chars=0`
- `degenerate_output=false`
- Required substrings matched: `PASS`, `42`, `def add_one`
- Required regex matched: `return\s+x\s*\+\s*1`
- Follow-up minimal torch XPU tensor check passed on devices `0`, `1`, `2`,
  and `3`.

This confirms the device reset recovered the non-EP TP4 graph path and that the
current promoted recipe still clears the semantic canary after the EP crash.

## Next Debug Targets

- Continue isolating whether free-form greedy token drift comes from XPU/oneCCL
  reduction order or other low-level numerical nondeterminism. Current probes
  already rule out async scheduling, XPU graph capture alone, and the llm-scaler
  INT4 MoE path as the sole cause.
- Debug the graph-mode shared-memory broadcast stall seen by short prompts after
  model load. Long-context graph quality and throughput runs still complete.
- Add optional finite-logprob/top-logprob capture to canaries when the runtime
  can return it without destabilizing XPU graph execution.

## MBT 1024 Stall

Tried raising `max_num_batched_tokens` from `512` to `1024` while keeping the
otherwise promoted TP4 full-decode graph recipe unchanged:

- Label: `full-decode-graph-triton-mbt1024`
- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-full-decode-graph-triton-mbt1024-tp4-ctx2048-mbt1024-bs256-p512n1536-20260514T155639Z-quality.json`
- Quality gate: `passed=true`, `nul_token_count=0`,
  `control_nonspace_text_chars=0`, `degenerate_output=false`
- Throughput log:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T155845Z.log`

Outcome: not promotable. The quality screen passed, but the throughput run
loaded the model, selected the llm-scaler INT4 MoE config, then repeated
`No available shared memory broadcast block found in 60 seconds` for more than
10 minutes before manual termination. The follow-up minimal torch XPU health
check then timed out, and `xpu-smi config -d 2 --reset` hung. `dmesg` showed Xe
PF/TLB invalidation errors during reset, so the system needs a full reboot
before more trustworthy benchmarks.

No LocalMaxxing submission was made. This records `max_num_batched_tokens=1024`
as a runtime-stability regression under the current graph recipe, despite a
clean semantic quality screen.

## Post-reboot Promoted Recipe Repeat

After rebooting from the `max_num_batched_tokens=1024` stall, the promoted
full-decode graph recipe was rerun with a shared-memory stall guard enabled:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-postreboot-promoted-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T163325Z-summary.json`
- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-postreboot-promoted-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T163325Z-quality.json`
- Quality gate: `passed=true`, `nul_token_count=0`,
  `control_nonspace_text_chars=0`, `degenerate_output=false`
- Output throughput repeats: `61.1757`, `60.8576` tok/s
- Mean output throughput: `61.0167` tok/s
- Mean total throughput: `81.3555` tok/s
- Follow-up minimal torch XPU tensor check passed on all four devices.

This is not a new LocalMaxxing submission because it reproduces the already
submitted promoted result rather than improving it. It does strengthen the
repeatability claim for the current quality-preserving recipe.

## Fast Piecewise Graph Recheck

The older faster piecewise graph/AOT recipe was rerun through the expanded raw
semantic canary suite before any throughput retest:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-fast-piecewise-graph-semantic-canary-20260514T164229Z.json`
- Runtime delta: `PIECEWISE` graph partition, `max_num_batched_tokens=1024`,
  old XPU graph cache root
  `/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-force-p512n256-20260513T101820Z`
- Result: `passed=false`
- Failure reasons: degenerate/corrupt output, required substring missing,
  required regex missing
- Corruption: `192` generated token IDs, all token id `0`/NUL,
  `distinct_generated_token_count=1`,
  `control_nonspace_text_chars=192`

This confirms the faster `~69` to `~73` tok/s piecewise/compiled/AOT lineage
remains invalid for quality. Do not use it as a headline benchmark until the
raw semantic canary suite passes.

## Timing Probe On Valid Path

Ran a short non-synchronized timing diagnostic on the promoted full-decode graph
path:

- Log:
  `/home/steve/bench-results/minimax-m2.7-timing/vllm-minimax-m27-autoround-tp4-p512n256-20260514T164927Z.log`
- JSON:
  `/home/steve/bench-results/minimax-m2.7-timing/vllm-minimax-m27-autoround-tp4-p512n256-20260514T164927Z.json`
- Shape: `512/256`, TP4, full-decode graph, `block-size=256`,
  `max_num_batched_tokens=512`
- Output throughput: `55.30` tok/s
- Total throughput: `165.90` tok/s

Caveat: this hook does not see most steady-state decode work once the valid
path is inside an XPU graph. It mostly records prefill/profile and uncaptured
Python-visible regions. The visible rank-0 hotspots were:

- `minimax.moe.experts_total`: `658.7 ms` total over `310` calls
- `minimax.attn.qk_norm`: `469.4 ms` total over `310` calls
- prefill-shaped hidden allreduce `(512,3072)`: `448.3 ms` total over `250`
  calls

Interpretation: useful next optimizations still point at MoE expert execution,
Q/K RMS scheduling, and TP communication, but a lower-level graph-captured trace
is needed before attributing exact steady-state decode percentages.

## Direct Q/K RMS Helper Screen

Tried enabling the default-off direct MiniMax Q/K RMS XPU helper against the
current valid full-decode graph path:

```bash
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=2048
```

Outcome: not promotable and not a benchmark datapoint.

- First attempt failed during worker startup with oneCCL/OFI
  `atl_ofi.cpp:376 send` / `Cannot assign requested address`.
- After cleanup and a successful four-device torch health check, the retry
  reached model load, then stalled before the quality gate produced JSON with
  repeated shared-memory broadcast warnings.
- The workers were terminated manually and a follow-up four-device torch health
  check passed.

Decision: keep the direct Q/K helper off for the current recipe. This screen
also showed that quality-stage hangs need the same guardrails as throughput
runs, so `scripts/run-minimax-quality-gated-candidate.sh` now writes a quality
log and wraps the quality stage in `QUALITY_TIMEOUT`.
