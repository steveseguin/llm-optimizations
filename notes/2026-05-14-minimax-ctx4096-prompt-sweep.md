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
