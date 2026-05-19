# MiniMax MoE WS Skip Redundant Contiguous Negative

Date: 2026-05-19

## Summary

This run tested a narrow exact-math cleanup in the MiniMax llm-scaler W4A16
work-sharing MoE path:

```bash
VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS=1
```

When the hidden state and router logits tensors are already contiguous, the
patch passes them directly to the MiniMax llm-scaler MoE custom op instead of
calling `.contiguous()` unconditionally. If either tensor is non-contiguous, it
falls back to the existing `.contiguous()` behavior.

This was intended to reduce framework-side overhead without changing model
math, routing, quantization, collectives, or power limits.

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

- Output tok/s samples: `87.853053`, `89.606353`, `89.000196`, `89.080940`
- Total tok/s samples: `117.137403`, `119.475137`, `118.666928`, `118.774587`
- Mean: `88.885135` output tok/s, `118.513514` total tok/s

The current promoted strict high remains:

- `88.927945` output tok/s
- `118.570593` total tok/s
- LocalMaxxing: `cmpco63q90052nw01ov1zxvwp`

This candidate is `-0.042809` output tok/s below the promoted mean. The
difference is small, but it did not beat the current strict high, so it is not
promoted and not submitted to LocalMaxxing.

## Decision

Do not promote. The patch is quality-safe and can remain as a default-off
diagnostic/cleanup switch, but it is not part of the current best speed recipe.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-ws-skip-redundant-contiguous-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T141946Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-ws-skip-redundant-contiguous-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T141946Z-quality`
- Local data: `data/minimax-m27-moe-ws-skip-redundant-contiguous-negative-20260519.json`
- Patch record: `patches/minimax-moe-ws-skip-redundant-contiguous-negative-20260519.md`
