# MiniMax No-Clone Retie On Current Baseline

Date: 2026-05-18

## Result

Retested `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` on top of the current promoted MiniMax logits-WS no-attention-delay baseline:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

The candidate passed the full strict quality gate, then produced:

- Run 1: `82.120315` output tok/s, `109.493753` total tok/s
- Run 2: `82.562695` output tok/s, `110.083593` total tok/s
- Mean: `82.341505` output tok/s, `109.788673` total tok/s

This is `-0.076%` output tok/s versus the current promoted no-attention-delay baseline of `82.404268` output tok/s and `109.872357` total tok/s. It is quality-safe but not faster, so it is not promoted and was not submitted to LocalMaxxing.

## Quality

All strict gates passed:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

The result is still useful: the no-clone compile/allreduce retie is not corrupting checked outputs under XPU graph replay, but it does not remove enough overhead to beat the current promoted recipe.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-no-clone-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T183449Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-no-clone-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T183449Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T185023Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T185313Z.json`

## Decision

Reject for performance. This should stay as a known quality-safe neutral result, not a promoted setting. The next useful path is still to remove or fuse decode boundaries rather than retie the same collective wrapper flags.
