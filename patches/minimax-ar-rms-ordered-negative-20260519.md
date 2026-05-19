# MiniMax Ordered AR+RMS XPU Patch Note, 2026-05-19

This records a rejected default-off MiniMax post-attention allreduce/RMS
experiment.

## Files

- `experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp`
- `experiments/minimax_ar_fused_rms_xpu/__init__.py`
- `benchmarks/b70_minimax_ar_fused_rms_compare.py`
- `vllm/model_executor/models/minimax_m2.py`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`

## Implementation

Added an extension op:

```text
minimax_ar_fused_rms_xpu::ar_rms(Tensor input, Tensor weight, str group_name, float eps) -> (Tensor, Tensor)
```

The model path is guarded by:

```bash
VLLM_MINIMAX_AR_RMS_XPU=1
```

When enabled, rank 0 adds the residual before the allreduce, matching MiniMax's
delayed residual allreduce math order, then the extension applies RMSNorm to the
allreduced tensor and returns that tensor as the residual.

## Result

Standalone microcheck was bit-exact against the delayed-reference path, but the
integrated model path was a performance negative:

- `raw145-n64-exact` passed.
- First generated run reported about `10.10` output tok/s.
- Inductor/ocloc hit `triton_red_fused__to_copy_mm_t_6` with `IGC: Internal
  Compiler Error: Floating point exception`.
- The screen was manually terminated during `raw145-n256-exact` loading.

## Decision

Do not promote and do not submit to LocalMaxxing. Keep this as a negative
reference: preserving delayed allreduce order is not enough if it moves the
current no-attention-delay baseline onto a slower hidden-state collective and
compiler path.
