# MiniMax Q/K Apply TP-Scale Rejection

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1` tested whether the MiniMax Q/K RMS helper should fold tensor-parallel variance scaling into the XPU apply kernel.

The previous clean path performs:

1. `minimax_qk_rms_xpu.var_alloc(qkv, q_size, kv_size)`
2. direct in-place `vllm.all_reduce_inplace(qk_var)`
3. `qk_var.mul_(1 / tp_world)`
4. `minimax_qk_rms_xpu.apply_alloc(...)`

This candidate keeps the exact same variance allreduce, but skips the separate `qk_var.mul_` kernel by passing `var_scale=1 / tp_world` into new `apply_scaled_alloc` / `apply_f32_weight_scaled_alloc` helper entry points. The kernel then computes:

```cpp
scaled_var = qk_var[token_idx * 2 + part] * var_scale;
scale = rsqrt(scaled_var + eps);
```

The feature is default-off and only enabled when `VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1`.

## Quality

Screen quality passed before benchmarking:

- `raw145-n64-exact`
- `raw145-n256-exact`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r8`

The exact raw145 token hashes matched the promoted references:

- n64: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- n256: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`

An XPU microcheck also confirmed bit-exact helper behavior for the scaled op:

- `apply_scaled_alloc(qkv, qk_var, ..., var_scale=0.25)`
- `apply_alloc(qkv, qk_var * 0.25, ...)`
- max Q diff: `0.0`
- max K diff: `0.0`

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.611230`, `88.107264`
- Total tok/s samples: `118.148306`, `117.476352`
- Mean: `88.359247` output tok/s, `117.812329` total tok/s

This is below the current clean direct in-place scale result:

- Current clean high: `88.501953` output tok/s, `118.002604` total tok/s
- Delta: `-0.16%` output tok/s

It is also below the warning-prone speed headline:

- Warning-prone high: `88.748424` output tok/s, `118.331232` total tok/s
- Delta: `-0.44%` output tok/s

## Decision

Reject for promotion.

The candidate is quality-safe in the screen gate, but the speed result is below the current clean high and the delta is too small to justify a longer full validation run. The separate `qk_var.mul_(1 / tp_world)` is not the bottleneck worth chasing in isolation.

Do not submit this result to LocalMaxxing. It is useful as a negative data point and as evidence that future Q/K work should fuse a larger boundary than only the scalar scale multiply.

## Reproduction

Key candidate env on top of the current clean MiniMax recipe:

```bash
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4
VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1
VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
CCL_TOPO_P2P_ACCESS=1
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
ZE_AFFINITY_MASK=0,1,2,3
```

Build note for the helper extension:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
export CC=/opt/intel/oneapi/compiler/2025.3/bin/icx
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MINIMAX_QK_RMS_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_QK_RMS_XPU_SYCL_DEVICE=bmg
pip install --no-build-isolation -e /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu
```

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-apply-tpscale-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T081116Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-apply-tpscale-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T081116Z-quality`
- Local data: `data/minimax-m27-qk-apply-tpscale-negative-20260519.json`
- Patch note: `patches/minimax-qk-apply-tpscale-negative-20260519.md`
