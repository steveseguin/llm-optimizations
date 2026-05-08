# 2026-05-08 MiniMax Next Steps

## Current State

Current valid MiniMax GGUF result:

```text
16.404929 tok/s, 4x Intel Arc Pro B70, MiniMax-M2.7 UD-IQ4_XS GGUF, p0/n64/r3
LocalMaxxing: cmowqyak0008co201oxuuzaid
```

This is a small improvement over the prior `16.383602 tok/s` K/Q/V-offload result. It adds SYCL RPC support for `GGML_OP_FUSED_MUL_UNARY`; same-build A/B was `16.404929` on versus `16.374820` off. Treat this as slight/near-noise, not a major bottleneck fix.

## Active Work

1. Keep the MiniMax AutoRound INT4 safetensors download running on `/mnt/corsair-external`.
2. When download completes, test vLLM/XPU TP4 with `Lasimeri/MiniMax-M2.7-int4-AutoRound`.
3. Continue using GGUF RPC+SYCL layer mode as the reproducible fallback while searching for a better all-GPU path.

## Bottleneck Hypothesis

Elementwise fused-op fixes are not enough. Fused RMSNorm is functional but neutral/slower; fused mul unary is only a tiny gain. The remaining gap versus DGX GB10-class MiniMax numbers is more likely in:

- attention/KV cache scheduling and copies
- MiniMax MoE routing/up-gate/down graph shape
- RPC layer-mode launch and graph scheduling overhead
- absence of a true efficient tensor-parallel/allreduce path for this huge GGUF model

## Next Experiments

1. vLLM AutoRound INT4:
   - Try TP4 on all B70s first.
   - If TP4 fails due quantized MoE/XPU support, capture the exact unsupported op and inspect vLLM/INC/AutoRound dispatch.
   - If TP4 OOMs or stalls, try reduced `max_model_len`, `max_num_batched_tokens`, and possibly TP2 just to isolate the failure mode.
2. GGUF attention/KV:
   - Add op timing around attention-side `CPY`, `ROPE`, `SOFT_MAX`, and `MUL_MAT` nodes.
   - Prefer producer-side fusion into KV writes over standalone copy kernels, because the tested MiniMax CPY fast path regressed.
3. GGUF graph split:
   - Revisit quality-correct graph reduce only if we can avoid host-mediated reduce/broadcast. The correct path works but is too slow.
   - Investigate device-side mirrored reduce for the exact nonlinear boundaries in MiniMax rather than broad deferred reductions.
4. System RAM:
   - Extra RAM will help reduce load/cache churn and make vLLM experiments less fragile, but it is not expected to fix decode throughput alone.
   - Keep using the USB disk for large model downloads/caches instead of pressuring NVMe free space.

