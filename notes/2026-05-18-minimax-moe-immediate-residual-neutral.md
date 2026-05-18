# MiniMax M2.7 Immediate MoE Residual Allreduce Retest

Date: 2026-05-18

## Summary

`VLLM_MINIMAX_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE=1` moved the post-MoE residual add into an immediate rank-0-residual allreduce path on top of the current exact logits-to-work-sharing MiniMax baseline:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, PIECEWISE XPU graph, exact MiniMax router logits feeding llm-scaler INT4 MoE work-sharing, `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- Shape: p512/n1536, ctx2048, batch 1, MBT512, block256

The candidate passed the full strict quality gate, but did not beat the promoted baseline:

- Candidate: `82.242981` output tok/s, `109.657309` total tok/s, mean of two gated benchmark repeats
- Promoted baseline: `82.404268` output tok/s, `109.872357` total tok/s, mean of four clean long repeats
- Delta: `-0.20%` output tok/s, `-0.20%` total tok/s

Decision: do not promote and do not submit to LocalMaxxing. The active runtime patch was reverted after the run.

## Quality Gate

The candidate passed:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic gate: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-immediate-residual-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T204631Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T210216Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T210511Z.json`

## Learning

This validates that a coarse Python/model-level relocation of the MoE residual collective can preserve outputs, but it does not remove enough real decode work to matter. Future work should target a true GPU-side fused epilogue or collective boundary, not another equivalent reshuffle in the model wrapper.
