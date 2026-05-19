# MiniMax MoE Full Forward Max2 Negative

Date: 2026-05-19

## Summary

This run kept the current strict MiniMax high-speed recipe but narrowed the
guarded MiniMax MoE full-forward custom-op boundary from decode-sized tensors
up to 4 tokens down to 2 tokens:

```bash
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=2
```

The intent was to test whether keeping token-2 graph/profile shapes inside the
custom-op boundary, while excluding wider shapes, improved repeatable decode
throughput versus the current max4 guard.

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

- semantic suite n64/r2: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- arithmetic repeat n64/r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64/r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

The extended-sixpack log printed `Bad address (src/pipe.cpp:367)` during
shutdown, but the strict runner continued to completion, `errors` remained
empty in the summary, and all four benchmark repeats completed. Treat this as
a runner/runtime warning to watch, not as a promoted performance result.

No quality-reducing changes were used: no speculative decoding, no expert
dropping, no router approximation, no quantization change, and no power-limit
change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `89.398809`, `88.351055`, `88.760240`, `88.905934`
- Total tok/s samples: `119.198413`, `117.801406`, `118.346987`, `118.541246`
- Mean: `88.854010` output tok/s, `118.472013` total tok/s

The current promoted strict high remains:

- `89.314195` output tok/s
- `119.085594` total tok/s
- LocalMaxxing: `cmpct6t4m007fnw01yjdtlcs4`

This candidate is `-0.460186` output tok/s (`-0.515%`) below the promoted
mean.

## Decision

Do not promote. Do not submit to LocalMaxxing. Keep
`VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=4` for the current speed
recipe.

The useful lesson is that max1 and max2 both preserve exact output quality but
do not outperform max4. The current full-forward custom-op boundary appears
best when it covers decode-sized tensors up to 4 tokens.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-fullforward-max2-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T181826Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-fullforward-max2-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T181826Z-quality`
- Local data: `data/minimax-m27-moe-full-forward-max2-negative-20260519.json`
