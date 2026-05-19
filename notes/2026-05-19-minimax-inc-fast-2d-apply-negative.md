# MiniMax M2.7 INC Fast 2D Apply Negative

Date: 2026-05-19

## Goal

Test whether `INCXPULinearMethod.apply()` could improve MiniMax decode speed by bypassing the generic `x.reshape(-1, x.shape[-1])` and final `out.reshape(out_shape)` path when XPU W4A16 linear input is already a 2D contiguous tensor.

The candidate was guarded behind `VLLM_XPU_INC_FAST_2D_APPLY=1` and tested on top of the current clean MiniMax TP4 recipe:

- `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`

## Quality

Quality passed all strict gates before benchmarking:

- raw145 n64 exact token hash
- raw145 n256 exact token hash
- semantic suite
- arithmetic repeat n64/r8
- extended sixpack n64/r2

This indicates the optimization is math-preserving for the tested graph path.

## Performance

Result, p512/n1536, ctx2048, TP4, batch 1:

- Output tok/s: `[87.60074345352838, 87.86610650032138]`
- Mean output tok/s: `87.73342497692488`
- Total tok/s: `[116.80099127137117, 117.15480866709518]`
- Mean total tok/s: `116.97789996923318`

Current clean promoted high remains `88.501953` output tok/s / `118.002604` total tok/s. The fast 2D apply candidate is therefore slower by about `0.87%` output tok/s.

## Decision

Reject and do not submit to LocalMaxxing. The candidate is quality-safe but slower than the current clean high.

The active source and venv copies were restored to the prior behavior after the test. Keep the patch only as a negative reference.

## Artifacts

- Summary: `data/minimax-inc-fast-2d-apply-full-repeat-20260519.json`
- Rejected patch note: `patches/minimax-inc-fast-2d-apply-negative-20260519.md`
- Full local summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-inc-fast-2d-apply-full-repeat-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T104059Z-summary.json`
