# MiniMax Post-Attention Norm Plus MoE Custom Op Negative

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_POST_ATTN_NORM_MOE_CUSTOM_OP=1` wraps the normal MiniMax
post-attention RMSNorm plus MoE call in a guarded custom-op boundary for
decode-sized tensors.

This stacked on the current strict high:

- `VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1`
- `VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`
- clone-safe compiled allreduce custom op
- llm-scaler INT4 W4A16 MoE work-sharing path

The candidate is default-off and guarded to XPU contiguous tensors with
`num_tokens <= 4`. It does not change RMSNorm math, MoE routing, expert
selection, quantization, sampling, speculative decoding, or power settings.

## Quality

Strict quality passed before benchmarking:

- `raw145-n64-exact`
- `raw145-n256-exact`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r16`
- `extended-sixpack-n64-r2`

The exact raw145 token hashes matched the promoted references:

- n64: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- n256: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`

Additional repeatability hashes:

- arithmetic repeat n64/r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64/r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `89.321280`, `88.826497`, `88.792850`, `89.087946`
- Total tok/s samples: `119.095040`, `118.435330`, `118.390467`, `118.783928`
- Mean: `89.007143` output tok/s, `118.676191` total tok/s

The current promoted strict high remains:

- `89.314195` output tok/s
- `119.085594` total tok/s
- LocalMaxxing: `cmpct6t4m007fnw01yjdtlcs4`

This candidate is `-0.307052` output tok/s below the promoted mean. The larger
custom-op boundary is exact, but it appears to hurt scheduling enough to lose
the smaller MoE-only boundary gain.

## Decision

Do not promote. Do not submit to LocalMaxxing. Keep
`VLLM_MINIMAX_POST_ATTN_NORM_MOE_CUSTOM_OP=0` for the current speed recipe.

The useful learning is that wrapping too much of the residual/RMS/MoE sequence
can be slower even when it is exact. The better direction is a narrower
boundary or a real fused XPU residual/RMS kernel that preserves the exact
operation order without blocking the favorable MoE scheduling.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-postattn-norm-moe-customop-plus-fullforward-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T163303Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-postattn-norm-moe-customop-plus-fullforward-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T163303Z-quality`
- Local data: `data/minimax-m27-postattn-norm-moe-customop-negative-20260519.json`
- Patch note: `patches/minimax-postattn-norm-moe-customop-negative-20260519.md`
