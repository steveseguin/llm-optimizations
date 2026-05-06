# 2026-05-06 llm-scaler source mining for B70 Q4 work

## Context

The user asked to keep `llm-scaler` on the TODO list. It may not be directly usable for Arc/B70 llama.cpp GGUF inference, but it is valuable source material because it contains Intel XPU/BMG ESIMD decode kernels and multi-GPU vLLM infrastructure.

Local clone:

`/home/steve/src/llm-scaler`

Current fetched state:

- commit: `e0b0703`;
- tag: `vllm-0.14.0-b8.2.1`;
- commit subject: `Update moe int4 prefill and decode (#384)`.

## Files Reviewed

- `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/int4_GEMV.h`
- `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/norm_gemv_int4.h`
- `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/resadd_norm_gemv_int4.h`

## Useful Ideas

`int4_GEMV.h`:

- explicitly targets decode GEMV on BMG XPU;
- treats GEMV at batch size 1 as the dominant latency contributor;
- varies `K_SPLIT` for small-N/high-K shapes so a single output row can use multiple work-items and SLM reduction;
- includes a fused multi-GEMV path for independent GEMVs sharing one activation vector, mainly to save launch overhead;
- documents launch overhead around `20-50 us` on BMG as a reason to fuse.

`resadd_norm_gemv_int4.h`:

- fuses residual add, RMSNorm, and INT4 GEMV;
- has a register-cached path for large `k_per_thread`;
- reuses the residual/norm work directly for the GEMV input;
- is a stronger conceptual match for llama.cpp graph-level work than another allreduce flag sweep.

## Caveat

The llm-scaler INT4 kernel comments call the format "GGML q4_0", but the implementation assumes group size 128 and scale shape `[N, K/128]`. llama.cpp GGUF `Q4_0` uses block size 32. Treat these files as BMG ESIMD scheduling/fusion references, not drop-in GGUF Q4_0 kernels.

## Impact on Current Plan

The Q4_0 four-card work is no longer limited by obvious communication toggles. The best 3x run is `46.194319 tok/s`; the best 4x assist split is `39.204149 tok/s`; equal 4x is worse at `34.929313 tok/s`.

The next software work should target:

- fewer per-token launches;
- fewer repeated quantization steps;
- fused Q4/Q8 decode GEMVs where multiple projections share the same activation;
- possible RMSNorm + GEMV fusion around Qwen layer boundaries;
- a controlled microbenchmark comparison between llama.cpp reordered MMVQ and an ESIMD Q4_0 x Q8_1 kernel before graph integration.

Do not spend more time on simple four-card split-ratio or allreduce-flag sweeps unless a profiler identifies a new narrow cause.
