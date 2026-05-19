# MiniMax Q/K Apply TP-Scale Candidate

Date: 2026-05-19

Status: rejected after quality-passed speed screen.

## Purpose

This candidate tried to remove one decode-sized scalar multiply kernel from the clean MiniMax Q/K RMS path by moving TP variance scaling into the XPU Q/K RMS apply helper.

It is default-off behind:

```bash
VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1
```

## Source Changes

`experiments/minimax_qk_rms_xpu/minimax_qk_rms_xpu.cpp`:

- Add `var_scale` to both apply kernels.
- Compute `scaled_var = qk_var[...] * var_scale` before `rsqrt`.
- Preserve existing `apply` / `apply_f32_weight` wrappers with `var_scale=1.0`.
- Add scaled exported entry points:
  - `apply_scaled`
  - `apply_scaled_return`
  - `apply_scaled_alloc`
  - `apply_f32_weight_scaled`
  - `apply_f32_weight_scaled_return`
  - `apply_f32_weight_scaled_alloc`
- Register the scaled entry points in pybind and Torch dispatch.

`experiments/minimax_qk_rms_xpu/setup.py`:

- Add `-D__DPCPP_SYCL_EXTERNAL_LIBC=__DPCPP_SYCL_EXTERNAL` so the extension rebuilds cleanly with oneAPI 2025.3 in this environment.

`vllm/model_executor/models/minimax_m2.py`:

- Add `VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE`.
- Require the scaled helper symbols when that flag is enabled.
- After direct in-place Q/K variance allreduce, set `qk_var_scale=1.0 / tp_world` instead of running `qk_var.mul_(1.0 / tp_world)`.
- Dispatch to the scaled apply helper when `qk_var_scale != 1.0`.

## Validation

Microcheck:

- Compared scaled helper against pre-scaled `qk_var` on XPU.
- Max Q diff: `0.0`.
- Max K diff: `0.0`.

Strict screen:

- raw145 n64 exact: passed.
- raw145 n256 exact: passed.
- semantic suite n64 r2: passed.
- arithmetic repeat n64 r8: passed.

Speed screen:

- p512/n1536, ctx2048, batch 1, TP4, 4x B70.
- Mean: `88.359247` output tok/s, `117.812329` total tok/s.
- Current clean high remains `88.501953` output tok/s.

## Decision

Keep this as a negative/marginal patch record, but do not promote it and do not submit it to LocalMaxxing.

The result indicates the separate tiny TP scale multiply is not worth optimizing by itself. A larger Q/K or residual boundary fusion is still the more promising path.
