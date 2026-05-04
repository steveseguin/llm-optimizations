# llama.cpp db44417 B70 SYCL Q8_1 activation cache patch

Date: 2026-05-04
Target checkout: `/home/steve/src/llama.cpp-q4-b70`
Primary file: `ggml/src/ggml-sycl/ggml-sycl.cpp`
Build tested: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench`

## Purpose

Qwen3.6 27B Q4_0 decode repeatedly quantizes the same F32 activation tensor to Q8_1 for sibling Q4_0 matmuls in a single graph compute. In tensor split, peer devices also repeatedly receive the same Q8 activation for those sibling matmuls.

This patch adds an opt-in graph-scoped cache for those Q8_1 activation buffers.

## Runtime switch

```bash
export GGML_SYCL_Q8_CACHE=1
```

`GGML_SYCL_Q8_CACHE=2` also asks the cache to print hit/miss counters through `GGML_LOG_INFO`, though in normal `llama-bench` runs the result artifacts are the more useful source of truth.

## Design

Added global cache state:

```cpp
int g_ggml_sycl_q8_cache = 0;

struct ggml_sycl_q8_cache_entry {
    const ggml_tensor * src1 = nullptr;
    const void *        src1_data = nullptr;
    int                 device = -1;
    int64_t             ne10 = 0;
    int64_t             nrows1 = 0;
    int64_t             src1_padded_col_size = 0;
    char *              ptr = nullptr;
    size_t              actual_size = 0;
    bool                ready = false;
};
```

Key fields are source tensor pointer, source data pointer, target device, and Q8 shape. The source data pointer is included so reused tensor objects cannot accidentally alias across storage changes.

The cache is cleared at the start and end of `ggml_backend_sycl_graph_compute_impl`, so entries are never reused across generated tokens/graph computes.

The cache is only active for the simple decode shape where `src1` is contiguous and `ne11 == ne12 == ne13 == 1`. Other shapes fall back to the original per-op allocation/quantization path.

## Hot-path changes

In `ggml_sycl_op_mul_mat`:

- allocate `src1_ddq` from the cache when `GGML_SYCL_Q8_CACHE=1` and the decode-shape guard passes;
- on the main device, skip `quantize_row_q8_1_sycl` when the cache entry is already ready;
- on peer devices in split mode, skip the Q8 peer copy when the peer cache entry is already ready;
- leave all non-cache cases on the existing path.

This preserves output quality because the cached bytes are exactly the Q8_1 activation produced by the existing quantization path for the same graph compute.

## Build

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j 4
```

## Bench commands

Single-card control/cache:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2 \
GGML_SYCL_Q8_CACHE=0 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -p 512 -n 256 -r 3 -ngl 99 -fa 0 -ub 128 -o jsonl
```

2-GPU tensor/cache:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,3 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1 -sm tensor -ts 1/1 \
  -p 512 -n 256 -r 3 -ngl 99 -fa 1 -ub 32 -o jsonl
```

3-GPU tensor/cache:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -sm tensor -ts 1/1/1 \
  -p 512 -n 256 -r 3 -ngl 99 -fa 1 -ub 32 -o jsonl
```

## Results

- 1x B70, 512/256: `24.425 tok/s` off, `24.500 tok/s` on.
- 2x B70, 512/256: `40.083 tok/s` off, `40.684 tok/s` on.
- 2x B70, 512/512 validation with cache: `40.487 tok/s`.
- 3x B70, 512/256: `40.937 tok/s` off, `42.432 tok/s` on.
- 3x B70, 512/512 validation with cache: `41.659 tok/s`.

## Caveats

- This is env-gated and should stay that way until broader prompt/batch shapes are tested.
- It does not address the single-card Linux-vs-Windows gap.
- It does not address the 4-GPU bottleneck, which is still dominated by many small cross-device reductions/synchronization points.
