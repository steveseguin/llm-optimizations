# MiniMax MoE Output Allreduce Inside Custom Op

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1` moves the final MiniMax MoE output tensor-parallel allreduce into the `vllm.moe_forward` custom-op boundary.

The guard is intentionally narrow:

- no shared experts
- no routed output transform
- no sequence parallelism
- TP or EP world size greater than 1
- fused output has not already been reduced
- delayed MoE allreduce is disabled

Under the same static guard, `_maybe_reduce_final_output()` skips the outer allreduce. This keeps the same math while moving the communication boundary closer to the MoE custom op and reducing one outer framework transition.

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

No quality-reducing changes were used: no speculative decoding, no expert dropping, no router approximation, no quantization change, and no power-limit change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.422654`, `89.595083`, `88.524703`, `89.169339`
- Total tok/s samples: `117.896872`, `119.460111`, `118.032938`, `118.892452`
- Mean: `88.927945` output tok/s, `118.570593` total tok/s

This is:

- `+0.48%` output tok/s versus the prior clean direct Q/K variance high (`88.501953`)
- `+0.20%` output tok/s versus the previous warning-prone speed headline (`88.748424`)

The gain is small, but it is the first strict-quality clean path here to beat the prior warning-prone speed headline.

## Runtime Recipe

Key env additions on top of the direct Q/K clean path:

```bash
VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4
VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0
```

Keep:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
```

## Decision

Promote as the current strict quality-passed MiniMax speed result.

LocalMaxxing accepted this result:

- ID: `cmpco63q90052nw01ov1zxvwp`
- Status: `APPROVED`

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-output-ar-inside-customop-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T125944Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-output-ar-inside-customop-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T125944Z-quality`
- Local data: `data/minimax-m27-moe-output-allreduce-inside-customop-20260519.json`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-moe-output-allreduce-inside-customop-p512n1536-20260519.payload.json`
- LocalMaxxing response: `data/localmaxxing-responses/minimax-m27-autoround-moe-output-allreduce-inside-customop-p512n1536-20260519.response.json`
- Minimal patch: `patches/minimax-moe-output-allreduce-inside-customop-20260519.patch`
