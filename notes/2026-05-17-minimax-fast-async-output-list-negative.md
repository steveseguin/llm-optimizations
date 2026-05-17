# MiniMax Fast Async Output-List Candidate

Date: 2026-05-17

## Result

The fast async output-list path is quality-safe on the exact canary, but it
does not improve throughput and should not be promoted.

Runtime flag:

```bash
export VLLM_XPU_FAST_ASYNC_OUTPUT_LIST=1
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
| adjacent control | 61.290142 | 81.720190 |
| fast async output list | 61.261073 | 81.681430 |
| delta | -0.029070 | -0.038760 |

Promoted strict baseline:

- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

Decision: do not promote and do not submit to LocalMaxxing. The candidate is
quality-safe on the first exact canary but is effectively neutral/slightly
slower than the adjacent control and below the promoted strict baseline.

## Artifacts

Result data:

- `data/minimax-m27-fast-async-output-list-negative-20260517.json`

Quality:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-output-list-screen/raw145-n64-fast-async-output-list-20260517T192714Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-output-list-screen/raw145-n64-fast-async-output-list-20260517T192714Z.log`

Benchmarks:

- control:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-output-list-control-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T193002Z.json`
- candidate:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-output-list-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T193305Z.json`

Bug caught during validation:

- The first canary attempt hit a `NameError` in the async output thread because
  the installed venv copy lacked the `timed_region` import. This was fixed
  before rerunning the passing canary.
- invalid log:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-output-list-screen/raw145-n64-fast-async-output-list-20260517T192336Z.log`

## Lesson

Per-token Python list construction is not the dominant bottleneck at this
point. The next useful target should be lower in the path: logits/local-argmax
collective timing, scheduler/output handoff, or a GPU-resident decode step that
removes a larger framework boundary.

No LocalMaxxing submission was made because this is a valid negative learning,
not an improved benchmark result.
