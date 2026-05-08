# 2026-05-08 MiniMax CPY Shape Trace

## Summary

Added a default-off SYCL worker trace:

```text
GGML_SYCL_CPY_TRACE=1
```

Then ran a MiniMax M2.7 `p0/n1` layer-mode pass to identify the actual hot `CPY` shapes before attempting a row-copy fast path.

Trace run:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/rpc-layer-cpytrace-r1-p0n1-20260508T084403Z.jsonl
```

## Observed Shapes

Aggregated across the four B70 RPC workers:

| Count | Shape |
| ---: | --- |
| 62 | `f32->f32 ne0=128 ne1=48`, source contiguous, destination row-strided, 24576 source bytes |
| 62 | `f32->f16 ne0=128 ne1=8`, source/destination contiguous, 4096 source bytes |
| 62 | `f32->f16 ne0=1 ne1=1024`, source column-view style, destination strided, 4096 source bytes |

The earlier same-type contiguous memcpy test could only affect the `f32->f32` case when the destination is also contiguous, which is not the hot MiniMax shape. The real work is f32-to-f16 conversion and row/column strided writes.

## Fast Path Test

Added a default-off shape-specific fast path:

```text
GGML_SYCL_CPY_MINIMAX_FAST=1
```

It covers the three traced shapes with simpler kernels:

- contiguous f32-to-f16 conversion;
- `ne0=1` strided f32-to-f16 conversion;
- row-strided f32-to-f32 copy.

Result:

```text
12.732294 tok/s, p0/n64/r1
```

This is a clear regression versus the current valid layer baseline and even versus noisy rebuilt off-runs. The custom kernels are not the right implementation. The likely issue is worse compiler/codegen behavior or occupancy versus the existing generic copy kernel, not math.

## Conclusion

The CPY bucket is real, but the simple shape-specific copy kernels are a dead end. Keep both trace and fast path default-off for reproducibility:

```text
GGML_SYCL_CPY_TRACE=0
GGML_SYCL_CPY_MINIMAX_FAST=0
```

A future copy optimization should either specialize inside the existing generic kernel more carefully or fuse the producer into the KV/cache write so the copy op disappears. A standalone replacement copy kernel is not enough.

