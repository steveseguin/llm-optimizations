# MiniMax Q/K Direct In-Place Scale

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1` adds a narrower call-site path inside the MiniMax Q/K RMS XPU helper.

For decode-sized Q/K variance tensors only (`float32`, `numel <= 2`), it calls the mutating no-return `vllm.all_reduce_inplace` custom op directly and then scales the variance in-place with `qk_var.mul_(1 / tp_world)`. Wider tensors keep the existing `tensor_model_parallel_all_reduce(qk_var) / tp_world` path.

This preserves the same math while bypassing the generic TP allreduce wrapper and one post-allreduce allocation for the hottest tiny Q/K variance collective.

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

No PyTorch custom-op alias warning was found in the quality or benchmark logs. One benchmark log printed `Bad address (src/pipe.cpp:367)` during shutdown after the benchmark JSON was written; this matches prior shutdown noise and is not treated as a quality failure.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.739272`, `88.302351`, `88.821529`, `88.144660`
- Total tok/s samples: `118.319029`, `117.736468`, `118.428705`, `117.526214`
- Mean: `88.501953` output tok/s, `118.002604` total tok/s

This is:

- `+0.21%` output tok/s versus the previous clean Q/K-helper path (`88.313105`)
- `+0.45%` output tok/s versus the alias-correct tiny-FP32 in-place baseline (`88.103866`)
- `-0.28%` output tok/s versus the faster warning-prone skip-clone speed headline (`88.748424`)

## Runtime Recipe

Key env additions on top of the current clean helper path:

```bash
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4
VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0
```

## Decision

Promote as the current clean-path MiniMax result.

It does not beat the `88.748424` warning-prone speed headline, but it is cleaner than that path and slightly faster than the prior clean helper result.

LocalMaxxing accepted this result:

- ID: `cmpc8cmqm0060pc016g5l5ukh`
- Status: `APPROVED`

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-direct-inplace-scale-20260519b-strict-tp4-ctx2048-mbt512-bs256-20260519T053656Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-direct-inplace-scale-20260519b-strict-tp4-ctx2048-mbt512-bs256-20260519T053656Z-quality`
- Local data: `data/minimax-m27-qk-direct-inplace-scale-20260519.json`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.payload.json`
- LocalMaxxing response: `data/localmaxxing-responses/minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.response.json`
- Minimal patch: `patches/minimax-qk-direct-inplace-scale-20260519.patch`
