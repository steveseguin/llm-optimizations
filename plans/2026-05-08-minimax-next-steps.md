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

1. MiniMax AutoRound INT4 safetensors are downloaded on `/mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround`.
2. Continue vLLM/XPU TP4 bring-up with `Lasimeri/MiniMax-M2.7-int4-AutoRound`. The first blocker was an unquantized XPU MoE fallback; the local experimental INC patch now routes MiniMax `FusedMoE` through WNA16 and fits the model at about 28.1 GiB/card. The current work item is proving generation after repairing vLLM package-version skew in the active environment.
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
   - Current model path: `/mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround`.
   - Preserve the local patch in `patches/vllm-inc-xpu-autoround-fusedmoe-wna16-20260508.patch`.
   - Use the repaired vLLM env at `/home/steve/.venvs/vllm-xpu` until the source checkout and installed package can be cleanly rebuilt.
   - Keep `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi` as the current best XPU env. `pidfd` produced `19.85` output tok/s and `99.231127` total tok/s at p512/n128, LocalMaxxing `cmox6tys30085ml0125gihg18`.
   - Compare `CCL_TOPO_P2P_ACCESS=1` and `0` if the model gets through load.
   - If TP4 fails in kernels, capture the exact unsupported op and inspect vLLM/INC/AutoRound dispatch.
   - Treat `--enforce-eager` as negative while compiled mode works; it regressed p64/n16 to `56.113901` total tok/s and `11.22` output tok/s.
   - Treat MiniMax QK-norm compile fusion as blocked on this XPU build. The flag is accepted, but `torch.ops._C.minimax_allreduce_rms_qk` is absent; the pass-manager crash was patched locally in `patches/vllm-minimax-qknorm-passmanager-xpu-guard-20260508.patch`.
   - Treat `VLLM_XPU_ENABLE_XPU_GRAPH=1` as negative for this TP4 path because vLLM disables graph capture around communication ops.
   - Treat `CCL_TOPO_P2P_ACCESS=0` as negative; p64/n16 with `pidfd` fell to `62.410028` total tok/s and `12.48` output tok/s versus `68.171339` and `13.63` with P2P=1.
   - Treat `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` as neutral diagnostic; p512/n128 reached `19.89` output tok/s versus `19.85` without the override, too small to promote.
   - Treat the AMD Instinct int4 W4A16 MoE tuned config as a negative seed for B70. The raw file failed on unsupported `matrix_instr_nonkdim`; stripping that key completed p64/n16 but regressed to `8.632747` total tok/s and `1.73` output tok/s.
   - Keep the hybrid B70 MoE config as the current best vLLM/XPU MiniMax setting. It uses tuned key `1` plus default prompt-size keys `64`, `256`, and `512`, and improved p512/n128 from `19.85` to `20.11` output tok/s (`100.538158` total), LocalMaxxing `cmox94fsm0095ml01tjeb20rr`.
   - Do not rely on the decode-only key `1` config by itself. It made the standalone MoE microbench slightly faster, but p64/n16 fell to `67.725172` total and `13.55` output tok/s because prompt shapes reused the decode config.
   - Retune larger prompt-size MoE configs only if microbench screening shows a stronger gain than default. The first B70 tune is a useful proof, but the remaining 30 tok/s gap is not just MoE tile selection.
   - Treat `--enable-expert-parallel` as negative/blocked for 4x B70 single-session MiniMax. Untuned EP fell to `3.75` output tok/s on p64/n16; the EP-specific MoE tune improved the standalone `E=64,N=1536` kernel but OOMed during model initialization with the tuned config.
   - Keep `MAX_BATCHED_TOKENS=1024` for p512/n128. Reducing it to `512` with the hybrid MoE config dropped output throughput to `13.57` tok/s.
   - If TP4 OOMs or stalls, try reduced `max_model_len`, `max_num_batched_tokens`, and possibly TP2 just to isolate the failure mode.
2. GGUF attention/KV:
   - Add op timing around attention-side `CPY`, `ROPE`, `SOFT_MAX`, and `MUL_MAT` nodes.
   - Prefer producer-side fusion into KV writes over standalone copy kernels, because the tested MiniMax CPY fast path regressed.
   - Keep `-fa 0` for current MiniMax GGUF runs. `-fa 1` aborts the SYCL/RPC worker with unsupported `FLASH_ATTN_EXT`; implementing that op or a safe fallback is a larger follow-up.
3. GGUF row packing:
   - Keep `GGML_SYCL_MMV_Y_RUNTIME=2` as the current MiniMax GGUF performance setting. A deterministic 16-token generation smoke matched default row grouping byte-for-byte.
   - Use `-ub 64`; it combined with fused RMSNorm to produce the current `17.697772 tok/s` high.
   - Keep fused RMSNorm enabled for now. The current r5 run and deterministic generation smoke both cleared.
   - Use `-sas 1` only as a local sweep option for now. It reached `17.718033 tok/s`, but the delta is too small for a separate public record.
   - Keep `-t 4` and `-rtr 1`; `-t 8` and `-rtr 0` did not improve.
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
