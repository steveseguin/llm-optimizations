# MiniMax Reusable Async Output-Copy Buffer Candidate

Date: 2026-05-17

## Result

The reusable pinned CPU sampled-token output-copy buffer is quality-safe on the
first exact canary, but it is slower than the adjacent control benchmark and
should not be promoted.

Runtime flags:

```bash
export VLLM_XPU_REUSE_ASYNC_OUTPUT_COPY_BUFFER=1
export VLLM_XPU_ASYNC_OUTPUT_COPY_BUFFER_SLOTS=3
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
| adjacent control | 61.626778 | 82.169037 |
| reusable buffer | 61.200924 | 81.601232 |
| delta | -0.425854 | -0.567805 |

Promoted strict baseline:

- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

Decision: do not promote and do not submit to LocalMaxxing. The first exact
quality canary passed, but the candidate is slower than both the adjacent
control and the promoted strict baseline.

## Artifacts

Result data:

- `data/minimax-m27-reuse-async-output-buffer-negative-20260517.json`

Quality:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-screen/raw145-n64-reuse-async-output-buffer-20260517T185414Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-screen/raw145-n64-reuse-async-output-buffer-20260517T185414Z.log`

Benchmarks:

- control:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-control-bench2/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T190757Z.json`
- candidate:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-v1-bench2/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T191102Z.json`

Invalid setup attempts:

- Missing strict interface pinning produced a candidate stall after AOT load:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T190100Z.log`
- Missing strict interface pinning also produced a control oneCCL ATL init
  failure:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/reuse-async-output-buffer-control-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T190652Z.log`

## Lesson

The sampled-token host handoff is still worth attacking, but this allocation
avoidance is too small and may add more bookkeeping than it removes. The
adjacent control also shows the current promoted path is repeatably around
`61.4` to `61.6` output tok/s, not the earlier unpromoted higher numbers.

Keep the reusable-buffer code default-off only as a reference. The next useful
optimization should reduce framework callbacks more aggressively: a GPU-resident
sampler/output handoff or a fused deterministic XPU local-argmax plus scheduler
handoff path.

No LocalMaxxing submission was made because this is a valid negative learning,
not an improved benchmark result.
