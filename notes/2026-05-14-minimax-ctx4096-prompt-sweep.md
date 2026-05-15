# MiniMax 4096 Prompt-Length Sweep

Goal: isolate the `max_model_len=4096` full-decode graph failure without
changing model weights, quantization, router precision, speculative decoding,
or power limits.

## Setup

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Runtime: vLLM `0.20.1-local`, XPU, TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Path: `CompilationMode.NONE`, `FULL_DECODE_ONLY` graph, `TRITON_ATTN`
- `max_model_len=4096`
- `max_num_batched_tokens=512`
- `block_size=256`
- `max_tokens=32`
- Prompt family: repeated PCIe/TP context body plus the same final question
- Token counts below are raw prompt tokens before the MiniMax chat template
  wrapper.
- Sweep tool: `scripts/run-minimax-ctx4096-prompt-sweep.py`

## Results

Summary:

`/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/summary-20260514T215840Z.json`

| Raw prompt tokens | Repeats | Result | NUL tokens | Wall time | JSON |
| ---: | ---: | --- | ---: | ---: | --- |
| 48 | 0 | pass | 0 | 120.88s | `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/ctx4096-mbt512-target48-actual48-20260514T215840Z.json` |
| 81 | 1 | pass | 0 | 130.00s | `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/ctx4096-mbt512-target80-actual81-20260514T215840Z.json` |
| 114 | 2 | pass | 0 | 129.95s | `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/ctx4096-mbt512-target110-actual114-20260514T215840Z.json` |
| 147 | 3 | fail: degenerate/corrupt output | 32 | 466.45s | `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/ctx4096-mbt512-target128-actual147-20260514T215840Z.json` |

The failing `147`-token point first sat in the shared-memory broadcast wait
path during init/profile, then completed graph capture and generated only token
id `0` / NUL for all `32` requested tokens.

The same `147`-token prompt shape failed in an earlier first-pass sweep as
well:

`/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/summary-20260514T215545Z.json`

## Interpretation

The boundary is much earlier than expected. Under `max_model_len=4096`, the
full-decode graph path is valid for a one-sentence prompt and for this repeated
context family through `114` raw prompt tokens, but becomes unstable at `147`
raw prompt tokens. The failure mode can be a long shared-memory wait followed
by all-NUL output rather than a clean `sample_tokens` timeout.

This points away from pure KV capacity as the first blocker. The immediate
suspects are the 4096-context graph profile/capture path, chunked-prefill
scheduling, or a prefill-to-first-decode replay boundary that starts failing
once the prompt crosses a small number of blocks for this prompt family.

## Next

- Add prompt-token and chunk diagnostics to the quality checker so every pass
  and failure records rendered chat-template token length, scheduled prefill
  chunks, and first decode shape.
- Re-run the boundary with `max_model_len=2048` as a control using the same
  prompt family. If `147` passes at 2048, the issue is specific to 4096 graph
  setup rather than the prompt content itself.
- Test 4096 graph with async scheduling disabled on the `114` and `147` points.
- Do not submit these results to LocalMaxxing; they are correctness/stability
  diagnostics, not benchmark achievements.

## Follow-Up: Q/K RMS Weight Corruption

Later finite tracing found a concrete corruption point: layer 58
`q_norm.weight` was finite during profile/warmup, then became partially NaN and
wildly out of range during real prompt execution. The checkpoint weights were
clean. This explains the all-NUL output as downstream NaN propagation rather
than a sampling-only bug.

The default-off clean-weight guard in
`patches/vllm-minimax-qk-rms-clean-weight-guard-20260515.patch` fixes the raw
prompt boundary for `max_model_len=2048` while keeping full-decode graph mode:

- `121` raw-token graph canary passed:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/graph-raw-neutral121-clean-weight-min2-ctx2048-n32-20260515T010220Z.json`
- Raw prompt gate with tokenizer-count `145` passed before a two-repeat
  throughput run:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-raw147-repeat-tp4-ctx2048-mbt512-bs256-p512n1536-20260515T011222Z-quality.json`

The original `max_model_len=4096` path still needs retest with the guard before
it can be promoted.

Retest completed:

- Quality-only graph retest passed:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep/graph-raw145-clean-weight-ctx4096-n64-20260515T012423Z.json`
- Full quality-gated throughput repeat passed:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-clean-weight-full-decode-graph-triton-ctx4096-raw145-repeat-tp4-ctx4096-mbt512-bs256-p512n1536-20260515T012656Z-summary.json`
- Mean throughput: `60.8976` output tok/s, `81.1968` total tok/s.
- LocalMaxxing: `cmp68w1mc00kso3016xc7jsgk`

The guard fixes the specific NUL-output prompt boundary at 4096 context. Broader
4096 validation remains open, especially longer prompts and shutdown hygiene
around oneCCL teardown warnings.
