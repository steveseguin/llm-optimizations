# MiniMax WS TopK FP16 Route-Weight Screen

Date: 2026-05-18

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM `0.20.1-local`, XPU TP4

Baseline for comparison: current strict promoted logits-to-work-sharing no-attention-delay recipe, `82.404268` output tok/s and `109.872357` total tok/s mean at p512/n1536, ctx2048, MBT512, block256.

## Summary

This candidate tested whether the exact MiniMax router-logits work-sharing path could use FP16 normalized route weights instead of FP32 route weights for the `moe_ws_down_cutlass_int4_kernel` specialization.

The routing decision stayed unchanged: the same top-8 experts were selected from sigmoid router scores plus MiniMax expert bias. Only the stored normalized route-weight dtype changed behind a default-off gate:

```bash
VLLM_XPU_MINIMAX_WS_TOPK_WEIGHT_FP16=1
```

This passed the full strict quality suite with the same promoted token hashes, but it was slower than the current promoted result.

## Candidate

```bash
VLLM_XPU_MINIMAX_WS_TOPK_WEIGHT_FP16=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
```

## Quality

All checks passed:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

This is important because FP16 route weights could have changed output ordering or repeatability. It did not, under this gate and benchmark shape.

## Benchmark

Shape: p512/n1536, ctx2048, batch 1, TP4, MBT512, block256.

Repeats:

- `80.511697` output tok/s, `107.348929` total tok/s
- `82.323588` output tok/s, `109.764785` total tok/s

Mean:

- `81.417643` output tok/s
- `108.556857` total tok/s

Delta vs current promoted baseline:

- Output tok/s: about `-1.20%`
- Total tok/s: about `-1.20%`

## Decision

Do not promote and do not submit to LocalMaxxing. The candidate is quality-safe, but it does not beat the current promoted no-attention-delay logits-to-WS result.

The useful learning is that route-weight dtype is not the next bottleneck in this path. The active work should stay on boundary fusion, graph scheduling, MoE/projection epilogues, and prefill/decode overlap rather than this FP32-to-FP16 route-weight substitution.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ws-topk-fp16-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T194905Z-summary.json`
- Benchmark 1: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T200505Z.json`
- Benchmark 2: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T200755Z.json`
- Build log: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T194720Z.log`
