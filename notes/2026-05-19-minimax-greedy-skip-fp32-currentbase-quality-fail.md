# MiniMax Greedy Sampler FP32-Skip Current-Baseline Quality Fail

Date: 2026-05-19

## Goal

Retest `VLLM_XPU_GREEDY_SKIP_LOGITS_FP32=1` on top of the current clean
MiniMax M2.7 4x B70 baseline:

- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4`
- `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`
- llm-scaler MiniMax W4A16 MoE work-sharing logits path
- clone-safe compiled allreduce custom op with tiny FP32 direct in-place Q/K variance scaling

This candidate skipped the sampler-side XPU logits-to-FP32 conversion for
guarded greedy/no-logprobs/no-processor requests and returned the greedy argmax
directly from the original logits dtype.

## Result

Label:

```text
minimax-greedy-skip-fp32-currentbase-20260519
```

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-greedy-skip-fp32-currentbase-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T122448Z-summary.json
```

Quality gate:

- `raw145-n64-exact`: passed
- `raw145-n256-exact`: failed exact combined token hash

Observed n256 facts:

```text
expected token hash: 58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537
observed token hash: e9e4aba8f7af253645a925ea8278df7a0e9a38154f379db96ea8fd5f13fc1f67
observed text hash:  8eaa0814c6f41fdb4bf0e679bed8c108bc80723dbc411ac6eabbcb4f334e27f3
distinct generated token count: 4
output speed shown by failed n256 run: 86.39 tok/s
```

No long p512/n1536 benchmark repeats were run because the strict runner only
benchmarks after all quality gates pass.

## Decision

Reject and do not submit to LocalMaxxing.

This same sampler idea was quality-clean but speed-negative on an older
`81.x` tok/s baseline. On the current `88.501953` clean baseline it changes the
longer exact raw145 output, so it is not quality-preserving under the current
compiled graph/runtime recipe. The active source and installed venv sampler
patches were removed after recording this result.

## Artifacts

- Data: `data/minimax-m27-greedy-skip-fp32-currentbase-quality-fail-20260519.json`
- Patch tried: `patches/minimax-greedy-skip-fp32-currentbase-quality-fail-20260519.md`
- Prior older-baseline screen: `notes/2026-05-18-minimax-greedy-skip-logits-fp32-negative.md`
