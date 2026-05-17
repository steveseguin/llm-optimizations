# MiniMax CCL Fabric-Vertex Override Screen

Date: 2026-05-17

## Summary

This candidate tested a pure oneCCL topology/runtime knob on top of the current
quality-promoted MiniMax M2.7 AutoRound local-argmax path:

- `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`
- `CCL_TOPO_P2P_ACCESS=1`
- `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
- `VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1`

The goal was to check whether oneCCL topology recognition was leaving B70
peer/fabric performance on the table. In normal runs, oneCCL reports PCIe-only
topology and suggests this override if fabric recognition is wrong. With the
override enabled, that warning disappeared, but decode throughput regressed.

## Quality Result

The candidate passed the full strict gate before benchmarking:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This makes the result useful as a clean performance negative rather than a
quality rejection.

## Benchmark Result

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Prompt/output/context: p512, n1536, ctx2048
- Batch/concurrency: 1
- Block size: 256
- Prefix cache: disabled
- Temperature: greedy / 0

Two repeats after the full strict quality gate:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 59.406462 | 79.208616 |
| 2 | 59.573265 | 79.431020 |
| mean | 59.489864 | 79.319818 |

Current strict baseline from `2026-05-17-minimax-localargmax-tightquality.md`:

| result | output tok/s | total tok/s |
| --- | ---: | ---: |
| strict baseline mean | 61.404035 | 81.872046 |
| this candidate mean | 59.489864 | 79.319818 |
| delta | -1.914171 | -2.552228 |

## Compiler/Driver Observation

The run repeatedly hit a recoverable Intel Triton/IGC compile failure during
piecewise graph capture:

- failed kernel: `triton_red_fused__to_copy_mm_t_10`
- shape in log: `xnumel = 256`, `r0_numel = 3072`
- command: `ocloc compile ... -device bmg`
- error: `IGC: Internal Compiler Error: Floating point exception`
- ocloc exit: `245`, build error `-11`

vLLM continued after the graph-capture failure and produced correct quality
hashes, so this is not a quality failure. It is still a driver/compiler issue
worth tracking because it may leave part of the graph uncaptured or on a slower
fallback path.

## Decision

Reject as a performance promotion. The override is quality-safe, but slower than
the current strict baseline by about `3.1%` decode throughput.

Do not submit this as a new LocalMaxxing row. It is a useful negative result:
the oneCCL fabric-vertex override should stay off for the current 4x B70 TP4
MiniMax path unless a future driver changes the behavior.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-ccl-fabric-vertex-override-localargmax-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T221412Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-ccl-fabric-vertex-override-localargmax-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T221412Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T222939Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T223237Z.json`

## Follow-Up

The next useful step is a same-day strict baseline confirmation with the same
driver/runtime state, followed by a closer look at the recurring `ocloc` ICE.
If the baseline still anchors near `61.4` output tok/s, then the next promotion
candidate needs to reduce actual decode work or TP synchronization, not just
adjust oneCCL topology assumptions.
