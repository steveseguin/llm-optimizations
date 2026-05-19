# MiniMax Q/K Apply TP-Scale Candidate

Date: 2026-05-19

Status: rejected. An older stack passed quality but was slower; a stricter
retest on the current promoted MoE-output-allreduce custom-op stack failed the
raw145 n256 exact token hash.

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

Strict retest on the current high:

- Candidate stack included `VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1`.
- `raw145-n64-exact`: passed.
- `raw145-n256-exact`: failed.
- Expected n256 token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`.
- Observed n256 token hash: `e9e4aba8f7af253645a925ea8278df7a0e9a38154f379db96ea8fd5f13fc1f67`.
- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-apply-tpscale-plus-moe-output-ar-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T145936Z-summary.json`.
- Benchmark was skipped.

## Decision

Keep this as a negative patch record, but do not promote it and do not submit it
to LocalMaxxing.

The result indicates the separate tiny TP scale multiply is not worth optimizing
by itself, and the current graph/custom-op stack can make the model-level result
non-exact even when the helper microcheck is bit-exact.
