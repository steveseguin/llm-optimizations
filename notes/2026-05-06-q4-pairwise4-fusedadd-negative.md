# Q4_0 Pairwise4 Fused-ADD Collective Probe

Date: 2026-05-06

## Summary

Tested a new 4-card pairwise tree path for the existing fused allreduce+residual-add helper in llama.cpp/SYCL.

The branch is gated by `GGML_SYCL_COMM_PAIRWISE4=1` and applies to `ggml_backend_sycl_comm_allreduce_add_tensor`, where Qwen3.6 27B Q4_0 spends most decode-time communication after reshape-through-ADD fusion.

Result: negative for decode. Keep the branch diagnostic-only and off by default.

## Build

- repo: `/home/steve/src/llama.cpp-q4-b70`
- build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- target: `llama-bench`
- model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
- quantization: `Q4_0`
- KV cache: `f16`
- no speculative decode
- no power-limit or clock changes
- patch artifact: `patches/llama-cpp-sycl-meta-pairwise4-fusedadd-current-20260506.patch.gz.b64`

## Common 4x Environment

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
export GGML_SYCL_DISABLE_DNN=1
export GGML_SYCL_Q8_CACHE=1
export GGML_SYCL_ASYNC_CPY_TENSOR=1
export GGML_SYCL_COMM_ALLREDUCE=1
export GGML_SYCL_COMM_SINGLE_KERNEL=1
export GGML_META_FUSE_ALLREDUCE_ADD=1
```

Common flags:

```bash
-dev SYCL0/SYCL1/SYCL2/SYCL3 -ngl 99 -sm tensor -ts 1/1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 128 -r 2 --poll 50 -o jsonl
```

## Results

| Run | Extra env | Prompt tok/s | Decode tok/s | JSONL |
| --- | --- | ---: | ---: | --- |
| Pairwise4 fused-add | `GGML_SYCL_COMM_PAIRWISE4=1`, `GGML_SYCL_COMM_EVENT_BARRIER=1` | 262.072535 | 32.707823 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-pairwise4-fuseadd-quad0123-p512n128-r2-20260506T004230Z.jsonl` |
| Same-build control | pairwise gate off, event barrier on | 102.228619 | 33.600765 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-pairwise4-control-gateoff-quad0123-p512n128-r2-20260506T004410Z.jsonl` |
| Pairwise4 no barrier | `GGML_SYCL_COMM_PAIRWISE4=1`, event barrier unset | 262.576268 | 33.009874 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-pairwise4-fuseadd-nobarrier-quad0123-p512n128-r2-20260506T004551Z.jsonl` |

Reference:

- prior clean 4x, `p512/n512/r1`: decode `34.375523 tok/s`;
- current best 3x, `p512/n512/r3`: decode `45.624065 tok/s`;
- post-test 3x health check, selector `2,1,3`, `p512/n128/r1`: decode `45.046330 tok/s`.

## Interpretation

The pairwise tree split did not reduce decode latency. It likely adds extra kernel/event overhead for the tiny 20 KiB f32 activation reductions and does not address the underlying count of reductions.

The pairwise prompt result is unusually high, but it is not actionable because decode regressed and llama-bench does not validate output correctness. Do not submit this result to LocalMaxxing.

Next work should target either:

- fewer repeated row-parallel allreduces;
- lower-cost collective data type experiments such as fp16 allreduce buffers with measured quality/error impact;
- speculative/MTP amortization of target-model collective overhead;
- ESIMD INT4 GEMV ideas from `llm-scaler`, especially if they can reduce per-token kernel launch or local matvec cost without hurting quality.
