# Qwen3.6 Q4_0 SYCL Event-Barrier Allreduce Diagnostic

Date: 2026-05-04

## Result

The event-barrier allreduce marker replacement is a quality-preserving Q4_0 improvement on the current three-B70 tensor-split path.

Validated command:

```bash
source /opt/intel/oneapi/setvars.sh --force

ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 \
  -sm tensor \
  -ts 1/1/1 \
  -p 512 \
  -n 512 \
  -r 3 \
  -ngl 99 \
  -fa 1 \
  -ub 32 \
  -ctk f16 \
  -ctv f16 \
  -t 8 \
  --poll 50 \
  -o jsonl
```

Validation output:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-eventbarrier-triple213-validate-p512n512-eb1-20260504T231752Z.jsonl`
- average decode: `43.605046 tok/s`
- samples: `43.9488`, `43.4813`, `43.385 tok/s`
- prompt average: `135.697394 tok/s`
- computed total throughput across separate prompt+decode legs: `65.999626 tok/s`

LocalMaxxing accepted the reduced public payload as `cmortp5vn000el404dj3zqv0u`.

## Patch

`GGML_SYCL_COMM_EVENT_BARRIER=1` only changes the existing experimental single-kernel allreduce path. After the root queue launches the full parallel peer-read allreduce kernel, non-root queues now receive an `ext_oneapi_submit_barrier({reduce})` dependency instead of submitting a tiny dependent `single_task` marker.

That keeps the allreduce math unchanged. It does not change model weights, KV dtype, sampling, speculative decode, or GPU power.

## Screen

Short `512/128` A/B with the same build and command shape:

| Setting | tok/s |
| --- | ---: |
| `GGML_SYCL_COMM_EVENT_BARRIER=0` | `41.983073` |
| `GGML_SYCL_COMM_EVENT_BARRIER=1` | `43.330808` |

## Interpretation

The previous small-F32 one-workgroup reduction lost badly because it underutilized the B70. This event-barrier variant keeps the full-range parallel kernel and only reduces queue marker overhead. The sustained 512-output validation improved from the prior Q4_0 TP3 Q8-cache validation of `41.659 tok/s` to `43.605 tok/s`.

Next useful Q4 screens:

- 2x B70 event-barrier validation;
- 4x B70 event-barrier screen, mainly to test whether marker overhead was part of the 4-card cliff;
- if 4x is still poor, move to fused matmul/allreduce epilogues or reduce-scatter-style graph changes instead of more marker tuning.
