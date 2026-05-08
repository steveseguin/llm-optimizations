# 2026-05-08 MiniMax Layer Knob And Kernel Screens

## Summary

After the quality-correct graph reduce diagnostic proved too slow, I tested lower-risk MiniMax layer-mode knobs and two small SYCL kernel ideas. None produced a new valid speed record.

Current valid reference remains:

```text
16.383602 tok/s, p0/n64/r3, -sm layer, -nkvo 0
LocalMaxxing: cmowft2hr000oo3019is4snoq
```

## Existing Runtime Knobs

Thread count sweep, p0/n64/r1:

| `-t` | tok/s |
| ---: | ---: |
| 1 | 16.254063 |
| 2 | 16.306743 |
| 4 | 16.279345 |
| 8 | 16.188375 |

Conclusion: RPC client CPU thread count is not the current bottleneck.

Flag failures:

- `-fa 1` is not usable in this MiniMax RPC+SYCL path. It sends `FLASH_ATTN_EXT` to the SYCL RPC worker, which aborts because that op is not implemented.
- Removing `GGML_DISABLE_FUSED_RMS_NORM=1` is not usable. The worker receives `FUSED_RMS_NORM` and aborts because that op is not implemented.

Candidate flag validation, p0/n64/r1:

| Variant | tok/s | Result |
| --- | ---: | --- |
| `-no-mmad 1` | 16.111982 | slower |
| `-fmoe 0` | 16.094589 | slower |
| `-no-mmad 1 -fmoe 0` | 16.166965 | slower |
| oneDNN enabled | 15.590120 | slower |

The short p0/n16 screen showed some warmed-run noise, but p0/n64 ruled out those toggles as improvements.

## SYCL Kernel Screens

Added a default-off same-type contiguous copy fast path:

```text
GGML_SYCL_CPY_MEMCPY_FAST=1
```

This only fires when source and destination have the same type, both tensors are contiguous, and byte sizes match. It is byte-identical when active.

Measured p0/n64/r1:

| Variant | tok/s |
| --- | ---: |
| fast path off | 15.546598 |
| fast path on | 15.591717 |

Conclusion: neutral within noise. This suggests MiniMax's hot `CPY` bucket is likely not dominated by same-type fully contiguous copies. A row-copy fast path may still be worth testing later, but the simple memcpy path is not enough.

I also tried an 8-expert `MUL_MULTI_ADD` unroll because MiniMax uses 8 selected experts. That regressed to `13.822999 tok/s`. More importantly, putting the unroll branch inside the device kernel can perturb the default path even when an env var is off, so I removed that patch rather than carrying a hidden default regression.

## Patch State

Current patch snapshot:

```text
patches/ik-llama-minimax-rpc-device-map-and-graphsplit-20260508.patch
```

The retained new code from this pass is only the default-off contiguous-copy fast path. The `MUL_MULTI_ADD` unroll was tested and removed.

## Next

The remaining likely layer-mode targets are not simple runtime toggles:

1. Implement a row-contiguous SYCL copy path for strided KV/cache copies, with explicit shape tracing first so the optimization matches the actual hot `CPY` shapes.
2. Improve `MOE_FUSED_UP_GATE` only after confirming whether the current direct merged `IQ4_XS` path is always taking the intended fast route.
3. Treat FlashAttention and fused RMSNorm as unsupported-worker gaps, not optimization knobs, unless we implement those ops in the SYCL RPC worker.
