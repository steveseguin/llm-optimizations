# Patch Notes: SYCL MUL_MAT Stage Stats and Selective Output DMMV

Date: 2026-05-04
Source tree: `/home/steve/src/llama.cpp-q4-b70`
Base commit reported by `llama-bench`: `db44417`
Primary file touched: `ggml/src/ggml-sycl/ggml-sycl.cpp`

## Purpose

This patch adds two diagnostic controls for Intel B70/Qwen3.6 Q4_0 work:

```text
GGML_SYCL_MUL_MAT_STATS=1
GGML_SYCL_DMMV_OUTPUT=1
```

`GGML_SYCL_MUL_MAT_STATS=1` is profiling-only. It adds explicit waits inside `ggml_sycl_op_mul_mat` so we can split timing into activation quantization, peer copies, matvec kernels, and destination copies.

`GGML_SYCL_DMMV_OUTPUT=1` is an experiment. It routes only the final `result_output` matmul through the DMMV path while leaving other Q4_0 matmuls on reordered MMVQ.

## High-Level Code Changes

Added env state:

```cpp
int g_ggml_sycl_dmmv_output = 0;
int g_ggml_sycl_mul_mat_stats = 0;
```

Added `GGML_SYCL_MUL_MAT_STATS` collection stages:

```cpp
enum ggml_sycl_mul_mat_stage {
    GGML_SYCL_MUL_MAT_QUANTIZE_MAIN = 0,
    GGML_SYCL_MUL_MAT_QUANTIZE_CHUNK,
    GGML_SYCL_MUL_MAT_PEER_COPY_Q8,
    GGML_SYCL_MUL_MAT_PEER_COPY_F32,
    GGML_SYCL_MUL_MAT_SRC1_COPY,
    GGML_SYCL_MUL_MAT_SRC0_COPY,
    GGML_SYCL_MUL_MAT_KERNEL,
    GGML_SYCL_MUL_MAT_DST_COPY,
    GGML_SYCL_MUL_MAT_SPLIT_WAIT,
    GGML_SYCL_MUL_MAT_STAGE_COUNT,
};
```

Inside `ggml_sycl_op_mul_mat`, stage timing wraps:

- `quantize_row_q8_1_sycl` for main-device activation quantization;
- Q8/F32 peer activation copies;
- non-contiguous source copies;
- the selected matmul op call;
- destination copies;
- split wait barriers.

At graph end:

```cpp
if (g_ggml_sycl_mul_mat_stats > 0) {
    ggml_sycl_mul_mat_stats_print();
}
```

Selective output DMMV dispatch:

```cpp
const bool prioritize_dmmv =
    g_ggml_sycl_prioritize_dmmv ||
    (g_ggml_sycl_dmmv_output && std::strcmp(dst->name, "result_output") == 0);

if (!prioritize_dmmv && should_reorder_tensor(...)) {
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
}
```

## Build Command

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j 2
```

Build completed successfully. The rebuild spent most of its time in `ocloc` BMG G31 AOT compilation and emitted existing register-spill warnings for unrelated kernels.

## Results

`GGML_SYCL_MUL_MAT_STATS=1` confirmed many small activation Q8_1 quantization launches per decode token.

Single B70 selector 2, one generated token:

```text
quantize_main: 345 calls, 284.483 ms explicit-sync total
mul_mat_kernel: 497 calls, 98.455 ms explicit-sync total
```

Forced global DMMV remains slower:

```text
15.518 tok/s over 64 generated tokens
```

Same-shape MMVQ control:

```text
22.683 tok/s over 64 generated tokens
```

Selective `result_output` DMMV is also slower:

```text
GGML_SYCL_DMMV_OUTPUT=1: 24.007 tok/s over 256 generated tokens
MMVQ control:            24.426 tok/s over 256 generated tokens
```

## Decision

Keep both switches diagnostic-only. Do not enable DMMV for submitted or production Q4_0 runs.

The next code direction should focus on launch-count reduction or graph scheduling around activation quantization plus reordered MMVQ, not DMMV replacement.
