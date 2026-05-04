# Qwen3.6 27B B70 Follow-Up: Q8_0 GGUF and Allreduce Timing

Date: 2026-05-04 UTC

Host: Ubuntu 24.04.4 LTS, 4x Intel Arc Pro B70 32GB. No GPU power-limit or clock changes were made.

llama.cpp worktree: `/home/steve/src/llama.cpp-q4-b70`, upstream `db44417` plus local experimental SYCL patches.

## FP8 / GGUF State

- Official FP8 model is already local at `/home/steve/models/qwen3.6-27b-fp8-hf`.
- That artifact is Hugging Face/Safetensors block-FP8, not GGUF.
- No native Qwen3.6 27B FP8 GGUF was found in the public manifests checked.
- Downloaded Q8_0 GGUF fallback: `/home/steve/models/qwen3.6-27b-q8_0-gguf/Qwen3.6-27B-Q8_0.gguf`.
- Q8_0 file size: `28,595,762,496` bytes.

## Q8_0 Results

Model file: `/home/steve/models/qwen3.6-27b-q8_0-gguf/Qwen3.6-27B-Q8_0.gguf`

| Setup | Prompt | Output | Result | Log |
| --- | ---: | ---: | ---: | --- |
| 1x B70, fit check | 0 | 1 | `15.146 tok/s` | `/home/steve/bench-results/qwen36-q8_0-gguf-sycl-single-fit-n1-20260504T132844Z.jsonl` |
| 1x B70 | 512 | 128 | `15.275 tok/s` decode | `/home/steve/bench-results/qwen36-q8_0-gguf-sycl-single-p512-n128-20260504T132932Z.jsonl` |
| 2x B70 tensor split | 512 | 128 | `25.733 tok/s` decode, `87.259 tok/s` total | `/home/steve/bench-results/qwen36-q8_0-gguf-sycl-dual03-p512-n128-20260504T133040Z.jsonl` |
| 3x B70 tensor split | 0 | 1 | aborts in allocation | `/home/steve/bench-results/qwen36-q8_0-gguf-sycl-triple213-fit-n1-20260504T133250Z.log` |
| 4x B70 tensor split | 0 | 1 | aborts in allocation | `/home/steve/bench-results/qwen36-q8_0-gguf-sycl-quad0123-fit-n1-20260504T133320Z.log` |

3x/4x Q8_0 abort with `ggml-backend.cpp:119: GGML_ASSERT(buffer) failed` from `ggml_backend_meta_alloc_ctx_tensors_from_buft()`. Treat this as a separate allocator/debug item. Q8_0 is usable as a higher-quality 1-2 GPU GGUF mode, but Q4_0 remains the speed target.

LocalMaxxing submission:

- Q8_0 2x B70: `cmor8w11d000lji04rn2zwh32`.
- The full annotated payload returned HTTP 500, so the accepted public submission contains the core metrics only. The full local payload is in `/home/steve/localmaxxing_payloads.json`.

## Allreduce Structure

Trace command shape:

```bash
GGML_META_ALLREDUCE_STATS=2 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -p 0 -n 1 -ngl 99 -dev SYCL0/SYCL1/SYCL2 \
  -sm tensor -ts 1/1/1 -fa 1 -ub 32 -r 1 -o jsonl
```

Trace output:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-meta-allreduce-detail-triple213-n1-stderr-20260504T131502Z.log`

Findings:

- 128 allreduces per decode graph.
- Every allreduce is `5120` F32 elements, `20480` bytes.
- Tensor names are one attention-output reduction plus one `ffn_out` reduction per layer.
- Recurrent layers use `linear_attn_out-N`; full-attention layers use `attn_output-N`.

These are residual-path boundaries, so they are not obviously removable by delaying through simple elementwise ops. This points toward lower-overhead communication, fused epilogues, or a different decomposition rather than more root-order sweeps.

## Timed Allreduce Diagnostics

`GGML_META_ALLREDUCE_STATS=3` synchronizes before and after each allreduce. This perturbs runtime and should only be used for diagnosis.

| Setup | Steady allreduce total per token | Avg per 20 KiB allreduce | Log |
| --- | ---: | ---: | --- |
| 2x B70 `0,3` | `1.718 ms` | `13.425 us` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-meta-allreduce-timing-dual03-n1-20260504T132711Z.log` |
| 3x B70 `2,1,3` | `5.732 ms` | `44.785 us` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-meta-allreduce-timing-triple213-n1-20260504T132429Z.log` |
| 4x B70 `0,1,2,3` | `10.605 ms` | `82.852 us` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-meta-allreduce-timing-quad0123-n1-20260504T132545Z.log` |

This confirms the 4-GPU regression is communication/synchronization overhead. The reduced tensor size is unchanged, but the per-reduction penalty rises sharply as the fourth GPU joins.

## Current Direction

- Keep Q4_0 as the primary speed path.
- Treat Q8_0 as a high-quality 1-2 GPU GGUF mode until TP3/TP4 allocation is fixed.
- Stop broad MMVQ launch-constant sweeps for now; `MMV_Y=2`, subgroup-8, and forced DMMV all regressed.
- Next useful Q4_0 work is either deeper MMVQ/dataflow optimization on single-card or a lower-overhead allreduce mechanism for multi-card.
