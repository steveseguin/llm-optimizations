# Qwen3.6 Q4 Runtime Recovery and 4x Confirmation

Date: 2026-05-05

## Context

After the MiniMax split `MUL_MAT_ID` prototype stalled, the first corrected Qwen3.6 Q4 run also stalled before any llama.cpp log output. `strace` showed the same `sched_yield()` spin pattern. Kernel logs showed `xe` device reset/coredump events:

- `0000:03:00.0`: GT0 engine reset and device coredump at 2026-05-04 20:41:13 local time.
- `0000:83:00.0`: GT0 schedule-disable failure, coredump, and reset at 2026-05-04 21:00:47 local time.

With no active compute processes, the runtime was recovered through debugfs GT resets:

```bash
sudo bash -lc 'for p in 0000:03:00.0 0000:83:00.0; do for gt in gt0 gt1; do echo 1 > /sys/kernel/debug/dri/$p/$gt/force_reset_sync; done; done'
```

OpenCL enumeration still showed all four B70s afterward.

## Recovery Smoke

Single-card Q4 smoke:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-recovery-smoke-single-p16n8-20260505T010215Z.jsonl
```

Result:

- prompt: `42.820 tok/s` for 16 prompt tokens;
- decode: `24.630 tok/s` for 8 output tokens.

This was only a runtime health check, not a benchmark submission.

## Corrected DNN-Off Screen

Common flags:

```bash
GGML_SYCL_DISABLE_DNN=1
GGML_SYCL_Q8_CACHE=1
GGML_SYCL_ASYNC_CPY_TENSOR=1
GGML_SYCL_COMM_ALLREDUCE=1
GGML_SYCL_COMM_SINGLE_KERNEL=1
GGML_SYCL_COMM_EVENT_BARRIER=1
```

Results:

| Case | Selector | Prompt tok/s | Decode tok/s | JSONL |
| --- | --- | ---: | ---: | --- |
| 3x tensor | `level_zero:2,1,3` | `135.588` | `42.170` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-eventbarrier-triple213-dnn0-p512n128-20260505T010300Z.jsonl` |
| 4x tensor | `level_zero:0,1,2,3` | `102.061` | `32.477` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-eventbarrier-quad0123-dnn0-p512n128-20260505T010415Z.jsonl` |

## Interpretation

The 4x Q4 regression is confirmed under the corrected DNN-off stack. It is not explained by the earlier accidental oneDNN-enabled subset sweep. The next Q4 work should target cross-GPU synchronization frequency or fusion, using 3x `43.605 tok/s` at 512/512 as the quality-preserving guardrail and 4x `32.4 tok/s` as the regression target.

This is not a new LocalMaxxing submission because the current 4x result is effectively a duplicate of the prior submitted negative TP4 datapoint.
