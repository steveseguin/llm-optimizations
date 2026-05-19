# MiniMax MoE Full Forward Plus Callable Cache Negative

Date: 2026-05-19

## Summary

This run stacked the current strict MiniMax high,
`VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1`, with the MiniMax llm-scaler MoE
callable cache:

```bash
VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1
```

The cache stores the selected MiniMax llm-scaler W4A16 logits work-sharing op on
each MoE layer during weight post-processing. Earlier tests showed it was
quality-safe but not a material win. This test checked whether it stacked with
the newer full-forward custom-op boundary.

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

Additional repeatability hashes also matched the current promoted high:

- arithmetic repeat n64/r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64/r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No quality-reducing changes were used: no speculative decoding, no expert
dropping, no router approximation, no quantization change, and no power-limit
change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.477853`, `89.028971`, `89.037226`, `88.771514`
- Total tok/s samples: `117.970470`, `118.705295`, `118.716301`, `118.362018`
- Mean: `88.828891` output tok/s, `118.438521` total tok/s

The current promoted strict high remains:

- `89.314195` output tok/s
- `119.085594` total tok/s
- LocalMaxxing: `cmpct6t4m007fnw01yjdtlcs4`

This cache stack is `-0.485305` output tok/s below the promoted mean. The
candidate is quality-safe, but it is a repeatable performance regression on top
of the full-forward custom-op stack.

## Decision

Do not promote. Do not submit to LocalMaxxing. Keep
`VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=0` for the current speed
recipe unless a future MoE boundary rewrite makes the cached callable relevant
again.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-plus-cache-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T160135Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-plus-cache-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T160135Z-quality`
- Local data: `data/minimax-m27-moe-full-forward-plus-cache-negative-20260519.json`
