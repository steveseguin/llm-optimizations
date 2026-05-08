# 2026-05-08 MiniMax Next Steps

## Current State

Current valid MiniMax GGUF result:

```text
17.697772 tok/s, 4x Intel Arc Pro B70, MiniMax-M2.7 UD-IQ4_XS GGUF, p0/n64/r5
LocalMaxxing: cmox103ol0040ml019yzs6gvs
```

This builds on the default-off `GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1` path, runtime MMV row packing with `GGML_SYCL_MMV_Y_RUNTIME=2`, `-ub 64`, and fused RMSNorm enabled. Same-build r5 default-MMV control was `17.198973 tok/s`; current stack is `17.697772 tok/s`. A synthetic IQ4_XS `MUL_MAT_ID` probe produced identical SYCL checksums and first outputs with fast MMID on versus off; a manual dequantized oracle showed the SYCL path is close (`nmse=1.44e-05`) while the CPU graph path diverges in this synthetic case.

The same stack at `p512/n128/r3` produced `54.506141 tok/s` prompt and `17.693021 tok/s` decode, LocalMaxxing `cmox1gcxl0049ml01kiijqbpo`. Decode throughput is essentially unchanged at 512 context, so the next MiniMax work should stay focused on decode-side matvec/MoE scheduling.

## Active Work

1. Keep the MiniMax AutoRound INT4 safetensors download running on `/mnt/corsair-external`.
2. When download completes, test vLLM/XPU TP4 with `Lasimeri/MiniMax-M2.7-int4-AutoRound`.
3. Keep `GGML_SYCL_MMV_Y_RUNTIME=2`, `-ub 64`, DNN disabled, and fused RMSNorm enabled as the default GGUF test setting; deterministic generation smokes matched earlier baselines byte-for-byte.
4. Investigate the CPU backend IQ4_XS `MUL_MAT_ID` mismatch against the manual dequantized oracle.
5. Continue using GGUF RPC+SYCL layer mode as the reproducible fallback while searching for a better all-GPU path.
6. Record negative GGUF kernel attempts when they rule out plausible optimizations.

## Bottleneck Hypothesis

Elementwise fused-op fixes are not enough. Fused RMSNorm is functional but neutral/slower; fused mul unary is only a tiny gain. Fast IQ4_XS `MUL_MAT_ID` plus MMV Y=2 row packing are useful, repeatable wins, but the remaining gap versus DGX GB10-class MiniMax numbers is more likely in:

- attention/KV cache scheduling and copies
- MiniMax MoE routing/up-gate/down graph shape
- RPC layer-mode launch and graph scheduling overhead
- absence of a true efficient tensor-parallel/allreduce path for this huge GGUF model

## Next Experiments

1. vLLM AutoRound INT4:
   - Try TP4 on all B70s first.
   - Start with llm-scaler-style XPU env: `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi`.
   - Compare `CCL_TOPO_P2P_ACCESS=1` and `0` if the model gets through load.
   - If TP4 fails due quantized MoE/XPU support, capture the exact unsupported op and inspect vLLM/INC/AutoRound dispatch.
   - If TP4 OOMs or stalls, try reduced `max_model_len`, `max_num_batched_tokens`, and possibly TP2 just to isolate the failure mode.
2. GGUF attention/KV:
   - Add op timing around attention-side `CPY`, `ROPE`, `SOFT_MAX`, and `MUL_MAT` nodes.
   - Prefer producer-side fusion into KV writes over standalone copy kernels, because the tested MiniMax CPY fast path regressed.
   - Keep `-fa 0` for current MiniMax GGUF runs. `-fa 1` aborts the SYCL/RPC worker with unsupported `FLASH_ATTN_EXT`; implementing that op or a safe fallback is a larger follow-up.
3. GGUF row packing:
   - Keep `GGML_SYCL_MMV_Y_RUNTIME=2` as the current MiniMax GGUF performance setting. A deterministic 16-token generation smoke matched default row grouping byte-for-byte.
   - Use `-ub 64`; it combined with fused RMSNorm to produce the current `17.697772 tok/s` high.
   - Keep fused RMSNorm enabled for now. The current r5 run and deterministic generation smoke both cleared.
   - Treat runtime/compile-time Y=4 and runtime Y=8 as neutral/negative for now. Compile-time MMV4 produced `17.191979 tok/s`; runtime MMV8 produced `17.238444 tok/s`.
   - Treat MoE-specific Y=4 as negative for now. `GGML_SYCL_MMV_Y_RUNTIME=2` plus `GGML_SYCL_MOE_IQ4_XS_MMV_Y=4` produced `17.232041 tok/s`.
4. GGUF graph split:
   - Revisit quality-correct graph reduce only if we can avoid host-mediated reduce/broadcast. The correct path works but is too slow.
   - Investigate device-side mirrored reduce for the exact nonlinear boundaries in MiniMax rather than broad deferred reductions.
5. System RAM:
   - Extra RAM will help reduce load/cache churn and make vLLM experiments less fragile, but it is not expected to fix decode throughput alone.
   - Keep using the USB disk for large model downloads/caches instead of pressuring NVMe free space.

## Negative/Neutral Results

- `GGML_SYCL_MOE_UP_GATE_PAIR_DOT=1`, paired IQ4_XS up/gate dot loop for MiniMax `MOE_FUSED_UP_GATE`: `16.840924 tok/s`, samples `15.8979`, `17.3159`, `17.3090`. This is neutral/slower than the `17.335655 tok/s` fast-MMID baseline, so it was not submitted to LocalMaxxing.
- Compile-time `GGML_SYCL_MMV_Y=4`: `17.191979 tok/s`, samples `16.5850`, `17.4923`, `17.4986`. This was not better than MMV Y=2 and was not submitted.
- Runtime `GGML_SYCL_MMV_Y_RUNTIME=8`: `17.238444 tok/s`, samples `16.6394`, `17.5203`, `17.5557`. This did not beat Y=2.
- Runtime `GGML_SYCL_MMV_Y_RUNTIME=2` plus `GGML_SYCL_MOE_IQ4_XS_MMV_Y=4`: `17.232041 tok/s`, samples `16.6258`, `17.5346`, `17.5357`. This did not beat generic Y=2.
- Flash attention `-fa 1` on the MiniMax 512/128 context run aborted worker 0 with unsupported `FLASH_ATTN_EXT`, so current valid runs stay on `-fa 0`.
- DNN-enabled RPC workers produced `17.213021 tok/s`, below DNN-disabled `-ub 64`; keep `GGML_SYCL_DISABLE_DNN=1`.
