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
- A later two-run quality attempt hung after model load, before graph capture.
  Treat this as a harness/runtime reliability issue to debug before requiring
  multi-run token-hash determinism.
- After fixing the wrapper JSON bug, a direct throughput rerun hung during
  distributed initialization after several interrupted experiments. It was
  killed and is not a benchmark datapoint.

## Current Policy

- The `61.75` output tok/s MiniMax result remains the current fastest
  quality-cleared speed datapoint because it has a clean long-context
  corruption smoke and tight throughput repeats.
- Do not claim token-exact deterministic decoding for this graph recipe yet.
- Future speed candidates must at minimum pass the one-shot long-context
  corruption gate and produce repeatable throughput. Stronger promotion should
  also resolve the two-run hash instability/hang or explain why exact token
  determinism is not expected on XPU graph.

## Next Debug Targets

- Isolate whether nondeterministic greedy tokens come from XPU graph capture,
  asynchronous scheduling, the no-op communicator graph-capture workaround, or
  XPU/oneCCL reduction order.
- Try a short canary prompt that should have a very high-margin deterministic
  next-token distribution, then compare graph versus eager tokens/logprobs.
- Run the same quality prompt with graph disabled and `CompilationMode.NONE` to
  establish whether eager TP4 is token-stable.
- Add a quality mode that records semantic canary substrings in addition to
  corruption checks, so exact free-form token hashes are not the only signal.
