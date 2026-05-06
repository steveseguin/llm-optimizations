# 2026-05-06 Q4_0 reordered MMVQ VDR4 screen

## Question

The standalone ESIMD Q4_0 harness gained a lot by block-loading scale metadata. Before porting a full ESIMD kernel into llama.cpp, test a lower-risk change in the active reordered subgroup MMVQ path: use one lane per Q4_0 block instead of two lanes per block.

The hypothesis was that `VDR=4` would remove duplicated Q4/Q8 scale loads in the current `VDR=2` path. The implementation is runtime-gated with `GGML_SYCL_REORDER_Q4_0_VDR4=1`; default behavior stays unchanged.

## Implementation

- Added a `reorder_vec_dot_q4_0_sycl_vdr<4>` functor.
- Taught the generic reordered MMVQ kernels to use the functor's `vdr_mmvq`.
- Added VDR4 dispatch for:
  - single Q4_0 reordered MMVQ;
  - fused2 Q4_0 reordered MMVQ;
  - fused2+SwiGLU Q4_0 reordered MMVQ.

## Command Shape

```bash
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=0 \
GGML_SYCL_FUSE_MMVQ2=1 \
GGML_SYCL_FUSE_MMVQ2_SWIGLU=1 \
GGML_SYCL_FUSE_RMS_NORM_MUL=1 \
GGML_SYCL_REORDER_Q4_0_VDR4={0,1} \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL2 -ngl 99 -sm none -fa 1 -ub 128 -ctk f16 -ctv f16 \
  -t 8 -p 0 -n 128 -r 2 --poll 50 -o jsonl
```

## Results

| Run | VDR4 | tok/s |
| --- | ---: | ---: |
| `p0/n1/r1` | off | 24.814015 |
| `p0/n1/r1` | on | 23.318414 |
| `p0/n128/r2` | off | 25.016114 |
| `p0/n128/r2` | on | 23.543413 |

## Decision

This is a clear regression. Keep `GGML_SYCL_REORDER_Q4_0_VDR4=0` and do not add it to any best-known recipe.

The result is still useful: the current subgroup kernel appears to prefer the existing two-lane-per-block scheduling despite duplicated scale loads. The ESIMD block-scale win likely requires the ESIMD row/tile load shape or a more direct layout-aware kernel, not just changing VDR in the current subgroup path.

Not submitted to LocalMaxxing because it is a negative experiment and not a useful leaderboard result.

