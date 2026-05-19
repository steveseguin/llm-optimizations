# MiniMax MoE Full Forward Max512 Negative

Date: 2026-05-19

## Summary

This run kept the current strict MiniMax high-speed recipe but raised the guarded
MiniMax MoE full-forward custom-op boundary from decode-sized tensors to
prefill/profile-sized tensors:

```bash
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=512
```

The intent was to see whether wrapping router-linear plus fused MoE across the
512-token prompt/profile path would reduce framework overhead without changing
model quality.

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

- Output tok/s samples: `85.682005`, `85.090115`, `85.146634`, `84.917575`
- Total tok/s samples: `114.242674`, `113.453487`, `113.528845`, `113.223433`
- Mean: `85.209082` output tok/s, `113.612110` total tok/s

The current promoted strict high remains:

- `89.314195` output tok/s
- `119.085594` total tok/s
- LocalMaxxing: `cmpct6t4m007fnw01yjdtlcs4`

This candidate is `-4.105113` output tok/s (`-4.596%`) below the promoted
mean.

## Decision

Do not promote. Do not submit to LocalMaxxing. Keep
`VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=4` for the current speed
recipe.

The useful lesson is that the decode-sized MoE custom-op boundary is beneficial,
but extending the same boundary into the 512-token prompt/profile path hurts
graph scheduling enough to erase the decode gain. Future prefill work should
target a separate prefill-specific path instead of broadening the decode path.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-max512-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T170714Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-max512-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T170714Z-quality`
- Local data: `data/minimax-m27-moe-full-forward-max512-negative-20260519.json`
