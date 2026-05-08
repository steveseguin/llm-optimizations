# 2026-05-08 MiniMax Fused RMSNorm SYCL Worker

## Summary

Implemented `GGML_OP_FUSED_RMS_NORM` in the SYCL RPC worker. Before this patch, enabling fused RMSNorm caused worker aborts:

```text
op not supported attn_norm-0 (FUSED_RMS_NORM)
```

The new worker path computes:

```text
y[j] = x[j] * rsqrt(mean(x^2) + eps) * weight[j]
```

This matches the CPU fused RMSNorm semantics for the supported f32 input / f32 weight case.

## Result

Smoke:

```text
p0/n1/r1: 3.018596 tok/s, completed
```

Speed screen:

```text
p0/n64/r1: 16.308342 tok/s, completed
```

This fixes the unsupported-op crash, but it is not a new speed record. The current valid MiniMax reference remains:

```text
16.383602 tok/s, p0/n64/r3, fused RMSNorm disabled
LocalMaxxing: cmowft2hr000oo3019is4snoq
```

## Repro

Worker env used for the fused RMSNorm test:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<id> \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
ZES_ENABLE_SYSMAN=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
rpc-server --device SYCL0 --host 127.0.0.1 --port <port>
```

Client env:

```bash
GGML_DISABLE_FUSED_MUL_UNARY=1
```

Do not set `GGML_DISABLE_FUSED_RMS_NORM=1` for this test.

## Interpretation

Fused RMSNorm support is still worth keeping because it turns a crash into a valid execution path. It just does not reduce enough kernel work to move MiniMax layer-mode throughput. The biggest remaining layer-mode costs remain attention matmuls, RoPE, KV/cache copy, softmax, and MoE up/gate/down combine.

