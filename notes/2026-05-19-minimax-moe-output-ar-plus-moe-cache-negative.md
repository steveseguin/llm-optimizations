# MiniMax MoE Output Allreduce Plus Callable Cache Negative

Date: 2026-05-19

## Summary

This run stacked the current strict MiniMax high,
`VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1`, with the earlier
quality-safe MiniMax llm-scaler MoE callable cache:

```bash
VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1
```

The callable cache stores the selected MiniMax llm-scaler W4A16 logits
work-sharing op on each MoE layer during weight post-processing. Earlier tests
showed it was quality-safe but too small/noisy to promote. This test checked
whether it stacked with the newer MoE output-allreduce boundary win.

## Quality

Strict quality passed before benchmarking:

- `raw145-n64-exact`
- `raw145-n256-exact`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r16`
- `extended-sixpack-n64-r2`

The exact raw145 token hashes matched the promoted quality references:

- n64: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- n256: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`

No quality-reducing changes were used: no speculation, no expert dropping, no
router approximation, no quantization change, and no power-limit change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `89.405693`, `88.933799`, `87.969720`, `89.339971`
- Total tok/s samples: `119.207590`, `118.578398`, `117.292960`, `119.119962`
- Mean: `88.912296` output tok/s, `118.549728` total tok/s

The current promoted strict high remains:

- `88.927945` output tok/s
- `118.570593` total tok/s
- LocalMaxxing: `cmpco63q90052nw01ov1zxvwp`

This stack is `-0.015649` output tok/s below the promoted mean. The difference
is tiny, but repeatable promotion requires beating the current strict mean, so
this result is rejected and not submitted to LocalMaxxing.

## Decision

Do not promote. Keep the callable-cache patch as quality-safe optional cleanup,
but not as part of the current speed recipe.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-output-ar-plus-moe-cache-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T134325Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-output-ar-plus-moe-cache-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T134325Z-quality`
- Local data: `data/minimax-m27-moe-output-ar-plus-moe-cache-negative-20260519.json`
