# MiniMax M2.7 Block128 Retest - 2026-05-17

Goal: retest `--block-size 128` after enabling the strict MiniMax-logits MoE path and local greedy argmax path, without weakening quality gates.

## Candidate

- Label: `minimaxlogits-localargmax-block128`
- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Runtime: vLLM 0.20.1-local, TP4, float16, XPU piecewise graph, Triton attention, llm-scaler INT4 MoE
- Key flags/env:
  - `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1`
  - `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
  - `VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
  - `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
  - `--block-size 128`
  - `--max-num-batched-tokens 512`
  - greedy temperature 0, prefix cache off

## Quality Gates

All strict gates passed:

- raw145 n64 exact hash
- raw145 n256 exact hash
- semantic suite n64/r2
- arithmetic repeat n64/r8
- extended sixpack n64/r2

Summary artifact:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-block128-strict-tp4-ctx2048-mbt512-bs128-20260517T075351Z-summary.json`

## Speed

p512/n1536, two repeats:

| Run | Total tok/s | Output tok/s | Artifact |
| --- | ---: | ---: | --- |
| repeat 1 | 81.605 | 61.204 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T080910Z.json` |
| repeat 2 | 82.121 | 61.591 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T081201Z.json` |
| mean | 81.863 | 61.397 | two-run mean |

## Decision

This is quality-safe, but it is slightly slower than the block-size 256 MiniMax-logits strict result of 61.464 output tok/s mean. Do not promote or submit this as a new LocalMaxxing achievement. Keep it as a repeatability and scheduler datapoint: block128 is valid, but block256 remains the best strict MiniMax-logits setting so far.
