# MiniMax Ordered AR+RMS XPU Negative

Date: 2026-05-19

## Goal

Revisit the rejected MiniMax post-attention allreduce-plus-RMS helper without
changing model quality. The prior `VLLM_MINIMAX_AR_FUSED_RMS_XPU=1` helper
allreduced the attention output first, then added the residual inside the RMS
kernel. That did not preserve the established token hashes across fresh runs.

This follow-up added a default-off ordered variant:

```bash
VLLM_MINIMAX_AR_RMS_XPU=1
```

The ordered variant preserves the delayed-residual math order: rank 0 adds the
replicated residual before allreduce, the extension allreduces that tensor, then
the extension applies RMSNorm and returns the reduced tensor as the next
residual.

## Microcheck

Added `benchmarks/b70_minimax_ar_fused_rms_compare.py` to compare both helper
orders against the delayed allreduce reference at the MiniMax hidden shape
`rows=1, hidden=3072`.

Result:

```text
old ar_fused_add_rms max_out_diff: 0.015625
old ar_fused_add_rms mean_out_diff: 0.001079559326171875
old ar_fused_add_rms max_residual_diff: 0.001953125
ordered ar_rms max_out_diff: 0.0
ordered ar_rms mean_out_diff: 0.0
ordered ar_rms max_residual_diff: 0.0
```

This confirms the previous helper had a real numerical-order mismatch, and the
ordered helper fixes that standalone issue.

## Model Screen

Initial strict model screen:

```text
LABEL=minimax-ar-rms-xpu-ordered-quality-screen-20260519
VLLM_MINIMAX_AR_RMS_XPU=1
VLLM_MINIMAX_AR_FUSED_RMS_XPU=0
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
BENCH_REPEATS=0
```

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ar-rms-xpu-ordered-quality-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T111713Z-summary.json
```

Observed:

- `raw145-n64-exact` passed with the promoted combined token hash.
- The first generation reported only about `10.10` output tok/s, far below the
  current clean `88.501953` output tok/s path.
- Inductor/Triton hit the familiar B70 `ocloc` internal compiler error on
  `triton_red_fused__to_copy_mm_t_6`, with `IGC: Internal Compiler Error:
  Floating point exception`.
- I manually terminated the follow-up during the `raw145-n256-exact` load rather
  than spending another model load on an obvious speed-negative path.

## Decision

Reject and do not submit to LocalMaxxing.

The ordered helper fixes the standalone math-order bug, but the integrated path
forces the slower delayed hidden-state allreduce shape and triggers the same
compiler-failure pattern seen in prior broad hidden-state allreduce experiments.
The next AR/RMS attempt should not move the current no-attention-delay baseline
onto the delayed rank-0 residual path. A useful version would need to fuse the
current `RowParallelLinear` reduce-results path with the following
`fused_add_rms_norm`, or lower the boundary in compiler/device code without
changing the no-delay numerical order.

## Artifacts

- Data: `data/minimax-m27-ar-rms-ordered-negative-20260519.json`
- Patch note: `patches/minimax-ar-rms-ordered-negative-20260519.md`
- Microcheck script: `benchmarks/b70_minimax_ar_fused_rms_compare.py`
