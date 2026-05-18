# MiniMax Q/K Helper In-Place Allreduce Negative

Date: 2026-05-18

## Result

Tested a localized Q/K RMS helper change on top of the current MiniMax logits-WS no-attention-delay baseline:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER_INPLACE_ALLREDUCE=1`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

The patch changed only the helper path's tiny Q/K variance collective from the generic tensor-parallel allreduce wrapper to an in-place `dist.all_reduce(qk_var, group=get_tp_group().device_group)` followed by in-place division. The goal was to keep exact math while avoiding wrapper clone/allocation overhead at a measured decode boundary.

The candidate passed the full strict quality gate but did not improve speed:

- Run 1: `81.124665` output tok/s, `108.166220` total tok/s
- Run 2: `82.753756` output tok/s, `110.338342` total tok/s
- Mean: `81.939211` output tok/s, `109.252281` total tok/s

This is `-0.56%` versus the current promoted no-attention-delay baseline of `82.404268` output tok/s. It is not promoted and was not submitted to LocalMaxxing.

## Quality

All strict gates passed:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

The quality result is useful: this confirms in-place `dist.all_reduce` on the Q/K variance tensor is not inherently corrupting the checked outputs under XPU graph replay. The speed result is the blocker.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-qk-helper-inplace-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T180746Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-qk-helper-inplace-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T180746Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T182325Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T182614Z.json`
- Patch sketch: `patches/minimax-qk-helper-inplace-allreduce-negative-20260518.patch`

## Decision

Reject for performance. The next Q/K path should not just replace the allreduce wrapper; it needs to remove or fuse a boundary, such as combining variance production, collective, and apply/rope scheduling more tightly, or using an existing vLLM communication fusion path if it can preserve exactness.
