# MiniMax MoE Full Forward Custom Op Plus Output Allreduce

Date: 2026-05-19

## Summary

`VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1` wraps the MiniMax MoE decode-sized router-linear plus fused MoE call in a MiniMax-specific `vllm.minimax_m2_moe_forward` custom-op boundary.

This stacks on the previous strict high:

- `VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`
- clone-safe compiled allreduce custom op
- llm-scaler INT4 W4A16 MoE work-sharing path

The new wrapper is default-off and narrowly guarded to XPU decode-sized contiguous tensors with `num_tokens <= 4`. It does not change routing math, expert selection, quantization, sampling, speculative decoding, or power settings.

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

Additional quality checks:

- arithmetic repeat combined token hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack combined token hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`
- deterministic across repeated runs
- no NUL/control-token degeneration

No quality-reducing changes were used: no speculative decoding, no expert dropping, no router approximation, no quantization change, and no power-limit change.

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4, 4x B70.

- Output tok/s samples: `88.927239`, `89.396677`, `89.527321`, `89.405544`
- Total tok/s samples: `118.569652`, `119.195570`, `119.369761`, `119.207393`
- Mean: `89.314195` output tok/s, `119.085594` total tok/s

This is a small but repeatable promotion:

- `+0.386250` output tok/s over the previous strict high (`88.927945`)
- about `+0.43%` output tok/s over the previous strict high
- `+0.515001` total tok/s over the previous strict high (`118.570593`)

## Runtime Recipe

Key addition:

```bash
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS=4
```

Keep the previous strict high settings:

```bash
VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4
VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0
VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
CCL_TOPO_P2P_ACCESS=1
```

## Decision

Promote as the current strict quality-passed MiniMax speed result. The gain is modest, but it survived the full quality gate and four long repeats.

LocalMaxxing accepted this result:

- ID: `cmpct6t4m007fnw01yjdtlcs4`
- Status: `APPROVED`

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-customop-plus-output-ar-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T151909Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-full-forward-customop-plus-output-ar-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T151909Z-quality`
- Local data: `data/minimax-m27-moe-full-forward-customop-plus-output-ar-20260519.json`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-moe-full-forward-customop-plus-output-ar-p512n1536-20260519.payload.json`
- LocalMaxxing response: `data/localmaxxing-responses/minimax-m27-autoround-moe-full-forward-customop-plus-output-ar-p512n1536-20260519.response.json`
- Patch note: `patches/minimax-moe-full-forward-customop-plus-output-ar-20260519.md`
