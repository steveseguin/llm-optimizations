# MiniMax Stream Interval 2048 Candidate

Date: 2026-05-17

## Result

The `--stream-interval 2048` scheduler option is quality-safe on the exact
canary, but it does not improve p512/n1536 batch-1 throughput on the current
MiniMax TP4 recipe.

Extra launch argument:

```bash
--stream-interval 2048
```

Common baseline flags:

```bash
export VLLM_XPU_LOCAL_ARGMAX_DECODE=1
export VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1
export VLLM_XPU_USE_LLM_SCALER_MOE=1
export VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
export VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
export CCL_TOPO_P2P_ACCESS=1
export FI_TCP_IFACE=wlxe865d47e3a48
export CCL_KVS_IFACE=wlxe865d47e3a48
```

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- p512 / n1536 / batch 1 / context 2048 / block size 256

Quality screen:

- raw145 n64 exact token hash passed
- expected and observed hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| adjacent control | 61.894320 | 82.525760 |
| stream interval 2048 | 61.404979 | 81.873305 |
| delta | -0.489341 | -0.652454 |

Promoted strict baseline:

- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

Decision: do not promote and do not submit to LocalMaxxing. The result is
quality-safe on the first exact canary, but the adjacent control was faster.

## Artifacts

Result data:

- `data/minimax-m27-stream-interval2048-no-improvement-20260517.json`

Quality:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/stream-interval2048-screen/raw145-n64-stream-interval2048-20260517T194409Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/stream-interval2048-screen/raw145-n64-stream-interval2048-20260517T194409Z.log`

Benchmarks:

- control:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/stream-interval2048-control-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T194954Z.json`
- candidate:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/stream-interval2048-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T194646Z.json`

## Lesson

Per-token streaming/output cadence is not the dominant bottleneck for this
single-request decode path. The next useful work should stay lower in the
critical path: collective timing, graph capture boundaries, MiniMax MoE routing
or logits path, and GPU-resident decode handoff.

No LocalMaxxing submission was made because this is a valid negative learning,
not an improved benchmark result.
