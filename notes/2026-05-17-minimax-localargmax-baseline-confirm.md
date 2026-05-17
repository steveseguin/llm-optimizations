# MiniMax Local-Argmax Baseline Confirmation

Date: 2026-05-17

## Summary

This run re-confirmed the current quality-safe MiniMax M2.7 AutoRound TP4
local-argmax recipe after the CCL fabric-vertex override screen. The goal was
to establish a same-day comparison anchor before testing further candidates.

Runtime:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Shape: p512, n1536, ctx2048, batch 1
- Block size: 256
- Prefix cache: disabled
- Temperature: greedy / 0
- `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
- `VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1`
- `CCL_TOPO_P2P_ACCESS=1`

## Quality Result

The run passed the full strict gate before benchmarking:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This confirms that the local-argmax recipe remains the current quality-safe
reference path.

## Benchmark Result

Three p512/n1536 throughput repeats after quality gates:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 60.672610 | 80.896814 |
| 2 | 60.550417 | 80.733889 |
| 3 | 60.703218 | 80.937624 |
| mean | 60.642082 | 80.856109 |

The previous promoted strict LocalMaxxing row was `61.404035` output tok/s and
`81.872046` total tok/s (`cmp9xpe3w04pdo4013acdikt7`). This confirmation is
about `1.24%` lower, but the three repeats are internally tight.

## Decision

Use `60.642082` output tok/s as the same-day comparison anchor for follow-up
candidate testing. Do not submit this as a new LocalMaxxing row because it is a
lower repeat of an already submitted recipe, not an achievement.

The practical conclusion is that small changes below about 1 tok/s are noise
unless they are repeated under the same strict quality gate. A real candidate
should beat this same-day anchor first and then be compared against the
previously published `61.404035` strict row before promotion.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-localargmax-baseline-confirm-20260517-strict-tp4-ctx2048-mbt512-bs256-20260517T223801Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-localargmax-baseline-confirm-20260517-strict-tp4-ctx2048-mbt512-bs256-20260517T223801Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T225330Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T225624Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T225911Z.json`

## Follow-Up

Next candidate: test the faster full-logits no-clone path with
`VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` and no local-argmax flag. This directly
checks whether final-hidden aliasing caused the prior no-clone quality drift.
