# MiniMax Q/K Helper Max512 Negative

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=512` was tested on top of the current alias-correct tiny-FP32 in-place allreduce path.

The intent was to apply the MiniMax Q/K RMS XPU helper during the 512-token prompt/profile path, not only decode-sized token counts. This was expected to improve prefill or total throughput without changing model math.

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

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `87.798918`, `88.574538`, `87.535096`, `87.988195`
- Total tok/s samples: `117.065224`, `118.099383`, `116.713462`, `117.317594`
- Mean: `87.974187` output tok/s, `117.298916` total tok/s

## Decision

Reject and do not submit to LocalMaxxing.

This is quality-safe, but slower than both current clean paths:

- Versus Q/K-helper decode-only clean path (`88.313105` output tok/s): `-0.38%`
- Versus direct in-place scale candidate (`88.501953` output tok/s): `-0.60%`

The useful lesson is that the Q/K helper should stay decode-sized for now. Applying it to the profile/prompt token range changes the compiled schedule and loses more than it gains.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-helper-max512-tinyfp32-inplace-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T050640Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-helper-max512-tinyfp32-inplace-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T050640Z-quality`
- Local data: `data/minimax-m27-qk-helper-max512-negative-20260519.json`
