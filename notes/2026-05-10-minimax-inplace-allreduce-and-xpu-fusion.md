# MiniMax In-Place Allreduce Screen and XPU Fusion Direction

## Summary

This pass tested whether vLLM's XPU compiled allreduce clone was a meaningful
MiniMax M2.7 decode bottleneck. It was not. The guarded in-place compiled
allreduce path was neutral to slightly negative and was reverted from the active
runtime after testing.

No model weights, quantization, prompt shape, speculative decoding, or GPU power
limits were changed.

## Results

| Label | Shape | Change | Cache / AOT | KV tokens | Output tok/s | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| `20260510T073135Z` | p512/n512 | fresh isolated default compile | `b5636051eb` / `c15860dd` | 9,408 | 27.23 | cold-compile artifact |
| `20260510T073455Z` | p512/n512 | same isolated default cache, warmed | `1d97049441` / `c15860dd` | 17,216 | 35.42 | current-floor repeat |
| `20260510T073806Z` | p512/n512 | `VLLM_XPU_ALLREDUCE_INPLACE_COMPILED=1`, fresh compile | `8b13139f79` / `c15860dd` | 9,408 | 26.95 | cold-compile artifact |
| `20260510T074110Z` | p512/n512 | same in-place-allreduce cache, warmed | `1daccf0db0` / `c15860dd` | 17,216 | 35.72 | neutral |
| `20260510T074339Z` | p512/n1536 | same in-place-allreduce cache, warmed | `1daccf0db0` / `c15860dd` | 17,216 | 36.69 | slightly negative |

The p512/n1536 in-place run trails the current default long-shape floor
(`37.05` output tok/s) and is far below the accepted `41.130667` output tok/s
LocalMaxxing high. It is not worth submitting.

## Interpretation

Fresh AOT compilation leaves enough device memory tied up during the first
profile/warmup path that KV cache capacity drops from `17,216` to `9,408`
tokens and throughput falls to roughly `27` output tok/s. The same AOT payload
is much better after restart/reload, so cold compile runs should not be used as
valid performance records.

Skipping the defensive `input_.clone()` in compiled XPU allreduce does not
recover the lost `39-41` tok/s AOT behavior. This matches the earlier XCCL
microbench: small allreduce payload latency alone is not the major remaining
ceiling. The more likely issue is the placement and synchronization of many
per-layer collectives inside the compiled graph.

## Next Direction

The next quality-preserving path is still a true XPU equivalent of vLLM's CUDA
MiniMax `minimax_allreduce_rms_qk` fusion:

- compute local Q/K variance from contiguous `qkv`;
- exchange the two per-token variance scalars across TP ranks through a
  persistent peer-visible workspace;
- apply global RMS scale to Q/K in the same kernel;
- eventually combine Q/K RMS apply with RoPE and KV-cache writes if the first
  fusion is correct and faster.

The CUDA implementation already uses an IPC/Lamport workspace. On XPU, the
equivalent building blocks appear to be Level Zero memory IPC and peer access:
`zeMemGetIpcHandle`, `zeMemOpenIpcHandle`, `zeMemCloseIpcHandle`,
`zeDeviceCanAccessPeer`, and `zeDeviceGetP2PProperties`. SYCL can access native
Level Zero context/device handles through `ext_oneapi_level_zero`, so the
prototype should start as a separate extension before being wired into vLLM's
`MiniMaxQKNormPass`.

Do not repeat more oneCCL algorithm toggles until there is a new hypothesis.
The measured XCCL microbench is already fast at the MiniMax decode payload
sizes; the model-level target is to remove or fuse graph-level collective
boundaries, not to tune the standalone collective.

