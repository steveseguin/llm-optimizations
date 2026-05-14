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

## Next Debug Targets

- Continue isolating whether free-form greedy token drift comes from XPU/oneCCL
  reduction order or other low-level numerical nondeterminism. Current probes
  already rule out async scheduling, XPU graph capture alone, and the llm-scaler
  INT4 MoE path as the sole cause.
- Debug the graph-mode shared-memory broadcast stall seen by short prompts after
  model load. Long-context graph quality and throughput runs still complete.
- Extend the raw semantic canary set beyond `PASS` with one arithmetic prompt
  and one short code prompt, both using pre-closed `<think>` raw templates.
- Add optional finite-logprob/top-logprob capture to canaries when the runtime
  can return it without destabilizing XPU graph execution.
