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
   - Treat `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` as neutral/negative. It was too small to promote on the earlier p512/n128 baseline (`19.89` versus `19.85` output tok/s), and it is slightly slower on the u4 decode path at p512/n256 (`32.726761` versus `33.033788` output tok/s).
   - Treat the AMD Instinct int4 W4A16 MoE tuned config as a negative seed for B70. The raw file failed on unsupported `matrix_instr_nonkdim`; stripping that key completed p64/n16 but regressed to `8.632747` total tok/s and `1.73` output tok/s.
   - Keep the hybrid B70 MoE config as the current best vLLM/XPU MiniMax setting. It uses tuned key `1` plus default prompt-size keys `64`, `256`, and `512`, and improved p512/n128 from `19.85` to `20.11` output tok/s (`100.538158` total), LocalMaxxing `cmox94fsm0095ml01tjeb20rr`.
   - Record the FP16 p512/n128 baseline as neutral/slightly positive: `20.17` output tok/s and `100.832219` total tok/s, LocalMaxxing `cmoxnvmna00gmml01eqdyl428`. This is not a major new ceiling because it changes activation dtype from BF16 to FP16.
   - Do not rely on the decode-only key `1` config by itself. It made the standalone MoE microbench slightly faster, but p64/n16 fell to `67.725172` total and `13.55` output tok/s because prompt shapes reused the decode config.
   - Retune larger prompt-size MoE configs only if microbench screening shows a stronger gain than default. The first B70 tune is a useful proof, but the remaining 30 tok/s gap is not just MoE tile selection.
   - Treat `--enable-expert-parallel` as negative/blocked for 4x B70 single-session MiniMax. Untuned EP fell to `3.75` output tok/s on p64/n16; the EP-specific MoE tune improved the standalone `E=64,N=1536` kernel but OOMed during model initialization with the tuned config.
   - Keep `MAX_BATCHED_TOKENS=1024` for p512/n128. Reducing it to `512` with the hybrid MoE config dropped output throughput to `13.57` tok/s.
   - Treat CPU `ngram` speculative decode as negative for this harness. It disabled async scheduling and reached only `2.26` output tok/s on p64/n16.
   - Treat GPU `ngram_gpu` speculative decode as negative for this harness. It kept async scheduling but reached only `3.15` output tok/s on p64/n16. Retesting `ngram_gpu` with the new u4 decode-only path at p512/n128 reached request processing, then worker processes terminated and vLLM reported `RuntimeError: cancelled`; no JSON throughput was produced.
   - Treat native MiniMax MTP as blocked with the current AutoRound checkpoint. The config advertises `use_mtp=True`, but the safetensors index has no layer 62-64 MTP tensors, and this vLLM tree has no MiniMax MTP adapter.
   - Promote the unsigned llm-scaler u4 decode-only variant as the current best MiniMax AutoRound software path. It keeps prefill on vLLM fused experts, routes only `x.shape[0] <= 4` FP16 decode batches through the custom raw-u4 ESIMD MoE kernel, and improves p512/n128 from the FP16 baseline `20.17` output tok/s to `29.74843` output tok/s (`148.742151` total). Removing the per-layer router-weight cast and leaving `CCL_ZE_IPC_EXCHANGE` unset improves p512/n256 to `34.578045` output tok/s, and p512/n512 reaches `37.136187` output tok/s. LocalMaxxing accepted the p512/n512 run as `cmoyagit0004dmk014gk25e2k`. Patch artifacts: `patches/llm-scaler-moe-int4-u4-decode-20260509.patch` and `patches/vllm-minimax-llm-scaler-u4-decode-20260509.patch`. Next useful work is a BF16-capable tiny kernel, reducing route/gather/top-k overhead in a more monolithic op, and profiling the remaining TP4 attention/allreduce decode cost.
   - Treat a thread-local tiny-MoE intermediate scratch-buffer cache as neutral/negative. It preserves math and only avoids one temporary allocation inside `moe_forward_tiny_cutlass_nmajor_int4_u4`, but p512/n256 reached `34.143113` output tok/s (`102.429339` total), below the best default-IPC u4 decode result of `34.578045`. The source and installed extension were restored to the previous allocation path after the test. Log: `vllm-minimax-m27-autoround-tp4-p512n256-20260509T174731Z.log`.
   - Keep `MAX_MODEL_LEN=2048` for current p512/n512 MiniMax AutoRound comparisons. `MAX_MODEL_LEN=1024` regressed p512/n256 to `24.269656` output tok/s; `MAX_MODEL_LEN=4096` regressed p512/n512 to `29.787984` output tok/s. vLLM reported only `0.56 GiB` available KV cache memory on both negative profiles versus `1.02 GiB` for the best 2048-token profile.
   - Treat decode-context parallelism as blocked on 4x B70 for this model. vLLM rejects `--decode-context-parallel-size 2` because TP4 is not greater than MiniMax's 8 KV heads for the GQA/MQA DCP path.
   - Treat benchmark wrapper changes as non-solutions for this p512/n512 path: `--async-engine` is neutral/slightly slower at `36.807084` output tok/s, and `--disable-detokenize` is neutral at `37.124066` output tok/s.
   - Treat XPU `--kv-cache-dtype fp8_inc` as blocked in this stack. It is also a quality tradeoff, and the current worker fails initialization with `Unsupported data type of kv cache: fp8_inc`.
   - Treat vLLM's built-in torch profiler as blocked for this TP4 XPU path. A p512/n32 profile run reached generation but repeatedly hit the shared-memory broadcast wait and emitted no trace files. Use narrower source-level counters/timers instead.
   - Treat `--attention-backend TRITON_ATTN` as negative for the current TP4 path. The p512/n256 screen selected Triton and ran, but dropped to `13.07` output tok/s and `39.222834` total tok/s. Keep default XPU FlashAttention.
   - Treat `--block-size 128` as negative for the current FlashAttention path. The p512/n256 screen dropped to `24.00` output tok/s and `72.014800` total tok/s. Keep the XPU FlashAttention preferred block size of `64`.
   - Treat forced XPU graph capture with communication as blocked in vLLM. `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1` and `XPU_GRAPH=1` reached `cudagraph_mode=PIECEWISE`, then failed during graph memory profiling because `parallel_state.py` asserts the tensor-parallel communicator is a `CudaCommunicator`; XPU/XCCL does not satisfy that path.
   - Treat the follow-up XPU graph workaround as negative for performance. A local patch allows `GroupCoordinator.graph_capture` to skip CUDA-only communicator capture for `XpuCommunicator` under `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`, and makes XPU skip the CUDA graph-memory profiler path that the local comment already marks unsafe for XPU. With `--kv-cache-memory-bytes 256M`, PIECEWISE graph capture completes and uses `1.15 GiB`, but p512/n256 reaches only `32.723015` output tok/s versus `34.578045` for the non-graph default-IPC path. LocalMaxxing accepted the diagnostic result as `cmoyfl7cm0057mk01suxo0glp`. Keep graph patches as a reproducibility artifact, not the current performance path.
   - Treat disabling prefix caching as neutral/slightly negative. `--no-enable-prefix-caching` reached `34.01` output tok/s and `102.041406` total tok/s at p512/n256, below the default prefix-caching result of `34.578045` output tok/s. Disabling chunked prefill with the current scheduler settings is blocked by vLLM validation, and vLLM warns MiniMax does not officially support manually disabling it.
   - Treat the opt-in FP16-router experiment as negative and quality-risky. `VLLM_MINIMAX_M2_FP16_ROUTER=1` precomputes FP16 gate weights and casts router logits back to FP32 for normal top-k, but p512/n256 dropped to `24.53` output tok/s and `73.576836` total tok/s. Keep the default FP32 router. Patch artifact: `patches/vllm-minimax-m2-fp16-router-experiment-20260509.patch`.
   - Standalone XCCL allreduce is not the obvious remaining ceiling: with explicit `CCL_ZE_IPC_EXCHANGE=pidfd`, MiniMax hidden-size allreduce probes are about `0.015 ms` for 6 KiB fp16, 12 KiB fp32, and 20 KiB fp32 payloads. The next profiling work should focus on attention/KV, router/top-k, custom MoE bridge overhead, and graph/scheduler gaps in context. Keep `pidfd` for standalone communication probes because default IPC hung before the first result.
   - A source-level `LLM_SCALER_MOE_TRACE_KERNELS=1` diagnostic shows tiny-MoE kernel waits are meaningful but not the whole decode ceiling. The p1/n4 trace produced 1753 wait samples with median `0.044650 ms`, average `0.057533 ms`, p95 `0.088551 ms`, and max `2.051210 ms`. One up+down launch pair is roughly `0.09 ms` per MoE layer at median timing, or about `5.5 ms/token` across the 62 MoE layers, versus about `26.9 ms/token` implied by the best p512/n512 result. Next instrumentation should time the Python/vLLM bridge, router/top-k, and attention/KV boundaries around the kernel calls.
   - A vLLM-level timing patch confirms the steady p512 decode step is mostly inside compiled model forward: p512/n8 rank-0 steady `runner.forward` is about `26.5-26.9 ms`, while preprocess/postprocess are sub-ms. An eager-only timing run is not a performance profile, but it exposes the likely shape: MiniMax Q/K norm and TP collectives are prominent (`qk_norm` p50 around `0.247 ms` per layer in eager, `tp.all_reduce.direct` p50 around `0.047 ms`, about three allreduces per layer). The existing vLLM MiniMax Q/K allreduce+RMS fusion is CUDA-only and `torch.ops._C.minimax_allreduce_rms_qk` is absent in this XPU build. A quality-preserving XPU equivalent is now a high-value target.
   - Treat the standalone XPU Q/K RMS helper as negative. It is numerically correct and graph-safe when registered as `torch.ops.minimax_qk_rms_xpu`, but it does not beat stock vLLM/Inductor: BMG AOT all-token p512/n256 reached `32.5460` output tok/s, and decode-only gating reached only `23.8984`, versus `33.4442` restored control and `34.5780` best baseline. Keep `VLLM_MINIMAX_QK_RMS_XPU_HELPER` unset. Artifact: `experiments/minimax_qk_rms_xpu/`; notes: `notes/2026-05-09-minimax-qk-rms-helper.md`.
   - Treat `VLLM_MINIMAX_QK_CONTIG=1` as negative. It makes packed-QKV q/k slices contiguous before exact Q/K allreduce+RMSNorm, but p512/n256 dropped to `33.92` output tok/s versus `34.578045` for the default-IPC baseline. Non-contiguous Q/K views are not the main bottleneck.
   - Keep the side-by-side oneAPI `2025.3.2` compiler installed for llm-scaler MoE-only builds. oneAPI 2026 hit a PyTorch bundled-SYCL header/runtime mismatch; the working MoE-only extension is built with 2025.3.2 and links to the vLLM venv's `libsycl.so.8`.
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
- vLLM CPU n-gram speculative decode for MiniMax AutoRound p64/n16 produced only `11.287082` total tok/s and `2.26` output tok/s. Not submitted.
- vLLM GPU n-gram speculative decode for MiniMax AutoRound p64/n16 produced only `15.728492` total tok/s and `3.15` output tok/s. Not submitted.
- Standalone `minimax_qk_rms_xpu` custom-op helper is negative. Raw pybind failed under TorchDynamo; registered custom ops work, but generic SPIR64 p512/n256 reached only `22.5659` output tok/s, BMG AOT reached `32.5460`, and decode-only BMG AOT reached `23.8984`. Keep disabled.
- Tiny-MoE intermediate scratch-buffer caching is neutral/negative. It restored successfully after testing and should not be carried as an active patch.
- oneCCL small-payload env tuning did not expose a hidden TP4 collective win. `CCL_MAX_SHORT_SIZE`, `CCL_PRIORITY=lifo`, and `CCL_WORKER_COUNT=1` were neutral to slightly negative around the 6-64 KiB MiniMax hidden-state allreduce sizes. Keep the current model-run defaults.
- A temporary MoERunner timing split shows router top-k is small (`~0.06 ms/layer`) compared with the routed expert apply path (`~0.18 ms/layer` around the llm-scaler bridge in decode). The active runtime was restored after measurement to avoid adding diagnostic context-manager overhead.
- The direct `torch.ops` u4 dispatch screen is not worth promoting. It was slightly faster in isolation, but p512/n256 was neutral/noisy (`33.62 tok/s` versus a restored `33.61 tok/s` control).
- The experimental unsigned ESIMD work-sharing u4 MoE kernel is a useful negative. It was much faster than scalar u4 in a standalone fake MiniMax layer, but regressed the real TP4 model to `31.65 tok/s` at p512/n256 and `35.39 tok/s` at p512/n512. Active vLLM is back on scalar `moe_forward_tiny_cutlass_nmajor_int4_u4`. Details: `notes/2026-05-09-minimax-comm-and-ws-moe-followups.md`.
- The first signed llm-scaler all-M vLLM integration is negative at full-context p512/n128: `12.27` output tok/s with `VLLM_XPU_USE_LLM_SCALER_MOE=1` versus `20.17` output tok/s for the FP16 baseline without it. Keep that old patch as a negative reproducibility artifact. The newer unsigned u4 decode-only gate supersedes it and is positive at `29.74843` output tok/s for p512/n128 and `33.033788` output tok/s for p512/n256.
