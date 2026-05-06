# 2026-05-06 Q4_0 allreduce + GET_ROWS fusion test

## Goal

Test whether the final remaining plain decode collective, `attn_output-63 -> GET_ROWS`, can be fused safely and whether that improves Qwen3.6 27B Q4_0 GGUF tensor-parallel decode on 3x B70.

## Patch

Added an off-by-default meta-backend/SYCL helper:

- env gate: `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1`;
- backend proc: `ggml_backend_comm_allreduce_get_rows_tensor`;
- path label: `backend+getrows`;
- exported by the SYCL backend only;
- correctness guard: f32 partials, i32 row indices, f32 output, mirrored `GET_ROWS` result, mirrored index tensor, `src_nbytes <= 64 KiB`, and output bytes no larger than source bytes;
- fallback: if the helper declines, run the normal allreduce and then compute the original `GET_ROWS` node.

Patch artifact:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-getrows-fusion-current-20260506.patch`
- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-getrows-fusion-current-20260506.patch.gz.b64`

## Trace result

Probe command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1 \
GGML_META_ALLREDUCE_STATS=4 \
llama-bench -m Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 0 -n 1 -r 1
```

The intended graph site was captured:

- `partial_probe`: `attn_output-63`, `MUL_MAT`, `nbytes=20480`, next op `GET_ROWS`, `next_nbytes=20480`;
- paths in the trace: `254` x `backend+add`, `2` x `backend+getrows`;
- no plain `backend` allreduce path remained for the traced decode graph.

Artifacts:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-probe-triple213-p0n1-20260506T000738Z.log`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-probe-triple213-p0n1-20260506T000738Z.jsonl`

## Timing results

Short screen, gate on:

- `p512/n128/r2`;
- prompt: `135.504744 tok/s`;
- decode: `45.129276 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-triple213-p512n128-r2-20260506T000850Z.jsonl`.

Full validation, gate on:

- `p512/n512/r3`;
- prompt: `135.626208 tok/s`;
- decode: `45.375471 tok/s`;
- computed total: `68.000508 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-triple213-p512n512-r3-20260506T001019Z.jsonl`.

Same-build control, gate off:

- `p512/n512/r3`;
- prompt: `135.671512 tok/s`;
- decode: `45.340867 tok/s`;
- computed total: `67.967329 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-control-gateoff-triple213-p512n512-r3-20260506T001228Z.jsonl`.

Reference high-water mark before this patch:

- post-reboot reshape-through-ADD 3x control: `45.624065 tok/s` decode, computed total `68.259384 tok/s`.

## Interpretation

The fusion is correct enough to capture the target site and run full validation, but it is neutral at best. The full run with the gate enabled was only `+0.034604 tok/s` over the same-build gate-off control and still below the current best `45.624065 tok/s` result.

Do not enable `GGML_META_FUSE_ALLREDUCE_GET_ROWS` for the default fast Q4_0 recipe yet. The useful lesson is that the remaining standalone `GET_ROWS` handoff is not a meaningful performance lever by itself; next Q4_0 work should target a lower-level fused row-parallel output epilogue or reduce the count/cost of the repeated 20 KiB reductions.

LocalMaxxing: not submitted. This is a neutral/negative implementation learning, not a leaderboard-quality improvement.
