# 2026-05-05 Q4 Allreduce-To-Reshape Experiment

## Goal

Remove the remaining plain allreduce cases that feed immediate `RESHAPE` nodes in the Qwen3.6 27B Q4_0 GGUF tensor-split graph on Intel Arc Pro B70.

Previous `GGML_META_ALLREDUCE_STATS=4` probing showed 48 `linear_attn_out-*` partial reductions per token with this shape:

`PARTIAL f32 MUL_MAT -> same-size mirrored RESHAPE`

The hypothesis was that a backend collective could reduce directly into the reshape output tensors and avoid a separate reshape/copy step.

## Patch

Added an experimental env-gated path:

- `ggml_backend_comm_allreduce_to_tensor`
- Meta backend detection gated by `GGML_META_FUSE_ALLREDUCE_RESHAPE=1`
- optional `GGML_META_FUSE_ALLREDUCE_RESHAPE_LIMIT`
- SYCL single-kernel implementation for 2-4 backends

This is intentionally not a default behavior.

## Trace

Command shape:

- model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
- selector: `level_zero:0,1,2,3`
- devices: `SYCL0/SYCL1/SYCL2/SYCL3`
- split mode: tensor, `-ts 1/1/1/1`
- `GGML_META_FUSE_ALLREDUCE_ADD=1`
- `GGML_META_FUSE_ALLREDUCE_RESHAPE=1`
- `GGML_META_ALLREDUCE_STATS=4`
- p0/n1

Files:

- JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-reshape-fuse-probe-quad0123-p0n1-20260505T052621Z.jsonl`
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-reshape-fuse-probe-quad0123-p0n1-20260505T052621Z.log`

Observed:

- `backend+reshape=96` across graph build/measure traces, meaning 48 intended paths per graph instance
- `backend+add=158`
- `backend_plain=2`, corresponding to the one remaining plain reduction per graph instance
- representative measured summary: 128 allreduces, `6.617 ms` total, `51.693 us` average

## Performance

4x B70, 512 prompt / 128 output:

- fused-add-only same-build control: `33.497463 tok/s`
- fused-add plus fused-reshape: `33.743952 tok/s`

3x B70 selector `2,1,3`:

- 512 prompt / 128 output, fused-add plus fused-reshape: `44.353959 tok/s`
- 512 prompt / 512 output, 3 reps, fused-add plus fused-reshape: `43.734996 tok/s`

Existing 3x fused-add-only validation remains better:

- `44.004344 tok/s` at 512 prompt / 512 output, 3 reps

## Decision

The graph recognition and backend dispatch are working, but this implementation is not a validated speed improvement. Treat it as an experimental negative/marginal result, not a LocalMaxxing submission.

Next useful direction is not more of the same direct-to-reshape kernel. Better candidates are:

- reducing root-device serialization in 4x collectives;
- true multi-root/pairwise collectives for these small 20 KiB reductions;
- reducing scheduling/copy overhead around tensor split rather than only fusing the reshape consumer.
