# 2026-05-08 MiniMax Fused Mul Unary SYCL Worker

## Summary

Implemented `GGML_OP_FUSED_MUL_UNARY` in the SYCL RPC worker and tested it on MiniMax M2.7 UD-IQ4_XS.

The supported worker paths are deliberately narrow:

- same-shape f32 `GELU`, `RELU`, and `SILU`
- row-gate f32 `SILU` and `SIGMOID`, where `src0->ne[0] == 1`

That matches the CPU fused-op semantics used by MiniMax without changing model weights, quantization, KV dtype, sampler behavior, or expert selection.

## Result

New current MiniMax r3 high:

```text
16.404929 tok/s, p0/n64/r3, 4x B70 RPC+SYCL layer split, F16 KV
LocalMaxxing: cmowqyak0008co201oxuuzaid
```

Same-build A/B:

| Variant | tok/s | samples |
| --- | ---: | --- |
| fused mul unary on | 16.404929 | 16.2231, 16.5070, 16.4847 |
| fused mul unary off | 16.374820 | recorded in `rpc-layer-disable-fused-mul-unary-r3-p0n64-20260508T095410Z.jsonl` |
| previous best | 16.383602 | 16.2585, 16.4391, 16.4532 |

Interpretation: this is a slight quality-preserving gain, but the delta is small enough to treat as near-noise rather than a major optimization. The patch is still useful because it closes another unsupported-op gap and did not regress the r3 decode run.

## Repro

Workers:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<id> \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
ZES_ENABLE_SYSMAN=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
rpc-server --device SYCL0 --host 127.0.0.1 --port <50100+id>
```

Client:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
llama-bench \
  -m MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 3 -ngl 99 -sm layer -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

Do not set `GGML_DISABLE_FUSED_MUL_UNARY=1` for the fused-op run.

## Next

This does not address the major MiniMax bottleneck. The useful next directions remain:

- test the AutoRound INT4 safetensors model in vLLM/XPU once the USB download completes
- inspect MiniMax attention and KV-copy hot paths, because fused RMSNorm and fused mul/unary only move small elementwise buckets
- keep the layer-mode RPC path stable as the reproducible GGUF fallback while searching for a better all-GPU tensor-parallel or vLLM path
