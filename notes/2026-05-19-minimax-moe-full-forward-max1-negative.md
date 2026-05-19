# MiniMax MoE Full Forward Max1 Negative

Date: 2026-05-19

## Summary

This run kept the current strict MiniMax high-speed recipe but narrowed the
guarded MiniMax MoE full-forward custom-op boundary from decode-sized tensors
up to 4 tokens down to a strict single-token decode path:

```bash
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=1
```

The intent was to see whether excluding graph-captured shape-2 decode/profile
paths would reduce dispatch overhead or avoid a slower compiled variant.

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

Additional repeatability hashes also matched the current promoted high:

- arithmetic repeat n64/r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64/r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No quality-reducing changes were used: no speculative decoding, no expert
dropping, no router approximation, no quantization change, and no power-limit
change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.317486`, `89.544082`, `89.334429`, `88.931577`
- Total tok/s samples: `117.756648`, `119.392109`, `119.112572`, `118.575437`
- Mean: `89.031893` output tok/s, `118.709191` total tok/s

The current promoted strict high remains:

- `89.314195` output tok/s
- `119.085594` total tok/s
- LocalMaxxing: `cmpct6t4m007fnw01yjdtlcs4`

This candidate is `-0.282302` output tok/s (`-0.316%`) below the promoted
mean.

## Decision

Do not promote. Do not submit to LocalMaxxing. Keep
`VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=4` for the current speed
recipe.

The useful lesson is that the current decode-sized custom-op guard is already
near the local optimum for this boundary. Reducing the guard to strict
single-token decode preserves exact output quality, but does not improve the
captured graph path or measured decode throughput.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-fullforward-max1-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T174710Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-fullforward-max1-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T174710Z-quality`
- Local data: `data/minimax-m27-moe-full-forward-max1-negative-20260519.json`
