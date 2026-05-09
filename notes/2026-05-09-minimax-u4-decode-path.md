# 2026-05-09 MiniMax AutoRound U4 Decode Path

## Result

The unsigned llm-scaler INT4 tiny-MoE path is now the best MiniMax AutoRound TP4 result:

| run | prompt/output | output tok/s | total tok/s | notes |
| --- | ---: | ---: | ---: | --- |
| FP16 vLLM baseline | 512/128 | 20.17 | 100.832219 | no llm-scaler path |
| signed llm-scaler all-M prototype | 512/128 | 12.27 | 61.374 | negative; prefill used tiny path |
| unsigned llm-scaler decode-only | 1/128 | 32.711775 | 32.967336 | decode isolation |
| unsigned llm-scaler decode-only | 512/128 | 29.74843 | 148.742151 | current best |
| unsigned llm-scaler decode-only | 512/256 | 33.033788 | 99.101363 | steady decode validation |
| unsigned llm-scaler decode-only + FP32 route weights | 512/256 | 34.157842 | 102.473527 | avoids per-layer router-weight cast |
| unsigned llm-scaler decode-only + PP2/TP2 | 512/256 | 17.550271 | 52.650812 | negative topology comparison |
| unsigned llm-scaler decode-only + FP32 route weights + default CCL IPC | 512/256 | 34.578045 | 103.734136 | current best |
| unsigned llm-scaler decode-only + default CCL IPC + `MAX_MODEL_LEN=1024` | 512/256 | 24.269656 | 72.808968 | negative compile-profile comparison |
| unsigned llm-scaler decode-only + FP32 route weights + default CCL IPC | 512/512 | 37.136187 | 74.272373 | best steady-state decode validation |
| unsigned llm-scaler decode-only + BF16 activations + default CCL IPC | 512/256 | 33.681326 | 101.043979 | quality-preserving BF16 validation |
| unsigned llm-scaler decode-only + BF16 activations + default CCL IPC | 512/512 | 36.607699 | 73.215399 | near-FP16 steady decode validation |
| unsigned llm-scaler decode-only + default CCL IPC + `MAX_MODEL_LEN=4096` | 512/512 | 29.787984 | 59.575969 | negative compile/KV-profile comparison |
| unsigned llm-scaler decode-only + default CCL IPC + async engine | 512/512 | 36.807084 | 73.614167 | neutral/slightly slower |
| unsigned llm-scaler decode-only + default CCL IPC + detokenize disabled | 512/512 | 37.124066 | 74.248133 | neutral; detokenization is not the bottleneck |
| unsigned llm-scaler decode-only + default CCL IPC + XPU PIECEWISE graph + fixed 256 MiB KV | 512/256 | 32.723015 | 98.169045 | graph capture succeeds but is slower |

Key log:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T015634Z.log
Throughput: 0.23 requests/s, 148.74 total tokens/s, 29.75 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T022204Z.log
Throughput: 0.13 requests/s, 99.10 total tokens/s, 33.03 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T105353Z.log
Throughput: 0.13 requests/s, 102.47 total tokens/s, 34.16 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp2-p512n256-20260509T111942Z.log
Throughput: 0.07 requests/s, 52.65 total tokens/s, 17.55 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T112853Z.log
Throughput: 0.14 requests/s, 103.73 total tokens/s, 34.58 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T113840Z.log
Throughput: 0.09 requests/s, 72.81 total tokens/s, 24.27 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T114811Z.log
Throughput: 0.07 requests/s, 74.27 total tokens/s, 37.14 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T193458Z.log
Throughput: 0.07 requests/s, 73.22 total tokens/s, 36.61 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T141049Z.log
Throughput: 0.13 requests/s, 98.17 total tokens/s, 32.72 output tokens/s
```

## What Changed

The earlier llm-scaler experiment converted packed INT4 weights to signed compact form and then used the tiny routed MoE path for both prompt and decode shapes. It proved decode speed but regressed full p512 because prefill also hit the tiny path.

The current patch does two narrower things:

- adds an unsigned uint4 variant in `llm-scaler` that decodes each raw nibble as `nibble - 8`;
- gates vLLM so the custom path only runs for MiniMax decode-size batches: `x.dtype == torch.float16` and `x.shape[0] <= 4`.
- accepts FP32 `topk_weight` tensors directly in the custom down-projection kernel so vLLM no longer casts router weights to FP16 for every decode-layer call.

This keeps prompt/prefill on vLLM's normal W4A16 fused-experts path and swaps only the decode MoE work onto the faster ESIMD kernel.

The `512/256` validation confirms the decode-side goal: when the fixed prefill cost is amortized over a longer generation window, single-session output throughput is above 30 tok/s. Removing the router-weight cast improved the same p512/n256 method from `33.033788` to `34.157842` output tok/s.

Leaving `CCL_ZE_IPC_EXCHANGE` unset so oneCCL can use its default IPC exchange improved p512/n256 again to `34.578045` output tok/s. The benchmark wrapper now supports `CCL_IPC=default` for that behavior; the older default remains `pidfd` unless explicitly overridden.

The p512/n512 validation with the same default-IPC path reached `37.136187` output tok/s. That is the best current steady-state MiniMax AutoRound decode result and suggests the fixed prompt/setup portion is still materially affecting shorter p512/n128 and p512/n256 comparisons.

The follow-up BF16 patch keeps MiniMax hidden states in BF16 while still using the llm-scaler u4 decode path. It fixes the earlier BF16 fallback from `16.860287` output tok/s to `33.681326` output tok/s at p512/n256, and reaches `36.607699` output tok/s at p512/n512. The focused write-up is `notes/2026-05-09-minimax-bf16-u4-decode.md`.

## Correctness Boundary

This is not speculative decoding and does not drop experts. No sampling parameters, router behavior, KV dtype, or GPU power limits changed.

The model quality boundary is still the selected AutoRound INT4 checkpoint. Relative to the previous AutoRound vLLM baseline, this patch changes only the MoE kernel/dequant path. The exact MiniMax MoE microbench measured max absolute difference around `3.052e-05` versus vLLM fused experts, and the raw-u4 decode path matched the signed-compact compatibility path exactly in the nibble conversion check.

The FP32 route-weight variant was smoke-tested on XPU with the same random tensors through FP32 and FP16 route-weight inputs; FP16 output max absolute difference was `1.9073486328125e-06`. That test only validates the route-weight input change, not end-to-end model quality.

## Speculative Decode Follow-Up

MiniMax `ngram_gpu` was retested with the new decode path:

```text
--speculative-config {"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_max":5,"prompt_lookup_min":2}
```

It reached request processing and then failed/stalled with worker termination and `RuntimeError: cancelled`; no JSON throughput result was produced. This reinforces the earlier CPU/GPU n-gram negative tests. For this MiniMax random-throughput harness, n-gram speculation is not the useful lever right now.

Native MTP remains blocked for this checkpoint: `config.json` advertises `use_mtp=true` and `num_mtp_modules=3`, but `model.safetensors.index.json` has zero `model.layers.62/63/64` or `mtp` tensors.

## Reproduce

Build the llm-scaler MoE-only extension with oneAPI 2025.3.2 and the vLLM XPU venv:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
cd /home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm
export PATH=/opt/intel/oneapi/compiler/2025.3/bin:$PATH
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
export LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib
export LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib:/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/torch/lib
rm -rf build python/custom_esimd_kernels_vllm/moe_int4_ops*.so
MAX_JOBS=2 TORCH_XPU_ARCH_LIST=bmg python setup_moe_int4_only.py build_ext --inplace -v
```

Run the benchmark:

```bash
USE_LLM_SCALER_MOE=1 \
DTYPE=float16 \
INPUT_LEN=512 \
OUTPUT_LEN=128 \
MAX_MODEL_LEN=2048 \
MAX_BATCHED_TOKENS=1024 \
MAX_NUM_SEQS=1 \
NUM_PROMPTS=1 \
TP=4 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Patch artifacts:

- `patches/llm-scaler-moe-int4-u4-decode-20260509.patch`
- `patches/vllm-minimax-llm-scaler-u4-decode-20260509.patch`

LocalMaxxing:

- `cmoxptkfd00hsml01hf2ajhhp`: p512/n128, `29.74843` output tok/s.
- `cmoxq7cww00i8ml019ihbeqc9`: p512/n256, `33.033788` output tok/s.
- `cmoy8hs3n002smk01ksgcpavr`: p512/n256 with FP32 route weights, `34.157842` output tok/s.
- `cmoy9exmf003lmk01d3it9cz2`: p512/n256 PP2/TP2 negative, `17.550271` output tok/s.
- `cmoy9qat60040mk01l5y8n3al`: p512/n256 with default CCL IPC, `34.578045` output tok/s.
- `cmoyagit0004dmk014gk25e2k`: p512/n512 with default CCL IPC, `37.136187` output tok/s.
- `cmoyfl7cm0057mk01suxo0glp`: p512/n256 with experimental XPU PIECEWISE graph and fixed 256 MiB KV, `32.723015` output tok/s; submitted as a negative/diagnostic learning result.

## Negative Follow-Up

`CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` was retested on the p512/n256 u4 decode path. It reached `32.726761` output tok/s and `98.180284` total tok/s, slightly below the default-topology `33.033788` output tok/s result. Keep default oneCCL topology recognition for this MiniMax path.

`VLLM_XPU_ENABLE_XPU_GRAPH=1` is not applicable to the TP4 path right now. vLLM logs this before benchmarking:

```text
XPU Graph doesn't support capture communication ops, disabling cudagraph_mode.
```

The run was stopped during shard loading after that warning because the engine config had already fallen back to `cudagraph_mode=NONE`.

`CCL_TOPO_P2P_ACCESS=0` was tested to switch oneCCL from P2P mode to USM mode. The setting took effect, and the run reached model load plus initial profiling/warmup, but then stalled before benchmarking with repeated:

```text
No available shared memory broadcast block found in 60 seconds.
```

The run was stopped without JSON throughput output. Keep `CCL_TOPO_P2P_ACCESS=1` for this MiniMax vLLM TP4 path.

PP2 x TP2 was tested as a four-GPU layout that reduces tensor parallel degree from 4 to 2 while keeping model memory around `28.0 GiB` per card. It fits and runs, but p512/n256 drops to `17.550271` output tok/s. For batch-1 single-session latency, pipeline bubbles and larger per-rank work dominate any reduction in TP collective degree. Keep TP4/PP1 for this model.

Reducing `MAX_MODEL_LEN` from `2048` to `1024` was also negative on the current best TP4/default-IPC path. The p512/n256 run dropped from `34.578045` to `24.269656` output tok/s even though the prompt plus output fits within 1024 tokens. vLLM compiled a `(1, 1024)` graph/range and reported only `0.56 GiB` available KV cache memory after loading; for this model, keep `MAX_MODEL_LEN=2048`.

Increasing `MAX_MODEL_LEN` from `2048` to `4096` was also negative for p512/n512. It produced `29.787984` output tok/s and `59.575969` total tok/s, with vLLM again reporting only `0.56 GiB` available KV cache memory. The current useful profile remains `MAX_MODEL_LEN=2048`, which reported `1.02 GiB` available KV cache memory and reached `37.136187` output tok/s.

`--decode-context-parallel-size 2` is blocked for this four-GPU MiniMax run. vLLM rejects the config before model load:

```text
tensor parallel size 4 must be greater than total num kv heads 8 when enable decode context parallel for GQA/MQA
```

`--async-engine` is neutral/slightly negative at p512/n512: `36.807084` output tok/s versus `37.136187` for the default LLM benchmark path. `--disable-detokenize` is neutral at `37.124066` output tok/s, so detokenization overhead is not the meaningful bottleneck.

`--kv-cache-dtype fp8_inc` is blocked on this XPU stack. vLLM accepts the argument and logs its accuracy warning, but worker initialization fails with:

```text
Unsupported data type of kv cache: fp8_inc
```

Built-in vLLM torch profiling was also attempted with XPU activities on a short p512/n32 run. It reached generation but repeatedly logged:

```text
No available shared memory broadcast block found in 60 seconds.
```

No profile trace files were emitted, and the run was stopped. The profiler is too disruptive for this TP4 XPU path as configured.

`--attention-backend TRITON_ATTN` is a clear negative on the p512/n256 screen. It selected the Triton backend, compiled successfully, then produced only `39.222834` total tok/s and `13.07` output tok/s:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T130109Z.log
Using Triton backend.
Throughput: 0.05 requests/s, 39.22 total tokens/s, 13.07 output tokens/s
```

Keep the default XPU FlashAttention backend for the current MiniMax path.

`--block-size 128` is also negative under the default FlashAttention backend. The same p512/n256 screen produced `72.014800` total tok/s and `24.00` output tok/s:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T130937Z.log
Throughput: 0.09 requests/s, 72.01 total tokens/s, 24.00 output tokens/s
```

Keep the XPU FlashAttention preferred KV block size of 64.

Forcing XPU graph capture with the local `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1` experiment is blocked before benchmark execution. The run reached `cudagraph_mode=PIECEWISE`, then failed during graph memory profiling:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T131754Z.log
FMHA sycl-tla kernels cannot be captured with XPU graphs, falling back to PIECEWISE graph mode on XPU platform.
AssertionError: assert isinstance(self.device_communicator, CudaCommunicator)
```

This is a vLLM graph-capture integration blocker for TP on XPU/XCCL, not just a launch flag issue.

A follow-up local graph experiment removed that first blocker by letting `GroupCoordinator.graph_capture` skip CUDA-only communicator capture for `XpuCommunicator` under `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`. The next blocker was memory: at normal automatic KV sizing, XPU graph profiling estimated `1.15 GiB` graph memory and left only `0.05 GiB` available KV memory, below the `0.12 GiB` required for `max_model_len=2048`.

The final graph experiment also made XPU follow the local code comment that says CUDA graph memory profiling should be skipped on XPU, and pinned KV cache manually:

```text
--kv-cache-memory-bytes 256M
```

That run captured PIECEWISE graphs successfully:

```text
Graph capturing finished in 9 secs, took 1.15 GiB
GPU KV cache size: 4,224 tokens
```

But it was slower than the non-graph default-IPC baseline:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T141049Z.log
Throughput: 0.13 requests/s, 98.17 total tokens/s, 32.72 output tokens/s
```

Do not promote XPU graph mode for this MiniMax path yet. The patch is still useful as a reproducibility artifact because it proves PIECEWISE graph capture can be made to run on XPU TP4, but the capture overhead/shape currently loses to normal compiled eager execution.

`--no-enable-prefix-caching` is neutral/slightly slower on p512/n256:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T132708Z.log
Throughput: 0.13 requests/s, 102.04 total tokens/s, 34.01 output tokens/s
```

Keep prefix caching enabled. Disabling chunked prefill together with prefix caching was blocked by vLLM validation at the current `MAX_BATCHED_TOKENS=1024`; raising that to satisfy validation is not attractive yet because vLLM warns MiniMax does not officially support manually disabling chunked prefill.

An opt-in FP16-router experiment was added behind `VLLM_MINIMAX_M2_FP16_ROUTER=1`. It materializes FP16 copies of the MiniMax replicated gate weights after model load and computes router logits with an FP16 GEMM, then casts logits back to FP32 for vLLM's normal top-k path. This is a speed/quality tradeoff probe because lower-precision router math can change expert selection.

The result is negative on p512/n256:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T133722Z.log
Throughput: 0.10 requests/s, 73.58 total tokens/s, 24.53 output tokens/s
```

Keep the default FP32 router. Patch artifact: `patches/vllm-minimax-m2-fp16-router-experiment-20260509.patch`.

Standalone XCCL allreduce probes with explicit `CCL_ZE_IPC_EXCHANGE=pidfd` are fast at the MiniMax hidden-state payload sizes:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/xccl-decode-size-allreduce-4xb70-pidfd-20260509T124512Z.log
minimax_hidden_fp16, 3072 elements, 6144 bytes: 0.015159 ms
minimax_hidden_fp32, 3072 elements, 12288 bytes: 0.015141 ms
small_20kb_fp32, 5120 elements, 20480 bytes: 0.014614 ms
```

This says raw standalone XCCL latency is not large enough by itself to explain a roughly `26.9 ms/token` p512/n512 decode step. The remaining bottleneck is likely embedded graph/scheduler gaps, attention/KV work, router/top-k overhead, or the custom MoE bridge/kernels in context, rather than simple allreduce bandwidth alone. The same standalone allreduce script hung with `CCL_ZE_IPC_EXCHANGE` unset/default, while vLLM default IPC still benchmarks correctly; keep using explicit `pidfd` for standalone communication probes.

A source-level timing hook was added to the llm-scaler MoE submit path behind `LLM_SCALER_MOE_TRACE_KERNELS=1`. The diagnostic p1/n4 run is not a throughput result because the hook waits on every submitted kernel, but it gives a useful bound:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n4-20260509T124828Z.log
1753 kernel wait samples
median wait: 0.044650 ms
average wait: 0.057533 ms
p95 wait: 0.088551 ms
max wait: 2.051210 ms
```

At median timing, one up plus one down tiny-MoE launch is roughly `0.09 ms` per MoE layer on a rank. Across MiniMax's 62 MoE layers that is around `5.5 ms/token` of kernel wait in isolation. That is meaningful, but it is still far below the approximately `26.9 ms/token` implied by the best p512/n512 result, so the next work should not assume MoE matvec alone is the entire remaining ceiling. The likely targets are now the per-layer bridge/router path, attention/KV, and graph/scheduler gaps around TP execution.

An additional vLLM timing patch adds opt-in decode timers behind `VLLM_XPU_DECODE_TIMING=1`. The first compiled p512/n8 diagnostic is not a throughput result because it prints every step, but it usefully separates vLLM scheduler overhead from compiled model work:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n8-20260509T143536Z.log
runner.forward steady decode: about 26.5-26.9 ms on rank 0
runner.preprocess steady decode: about 0.3-0.5 ms
runner.postprocess steady decode: about 0.18-0.19 ms
moe.llm_scaler_u4_bridge steady calls: about 0.018-0.022 ms
```

This confirms the main remaining ceiling is inside the compiled model forward, not request scheduling or output postprocess.

## Compiled Timing Summary Follow-Up

I added an opt-in atexit summary to the timing helper so future profiling can
summarize rank-local timing without printing every layer call:

```text
VLLM_XPU_DECODE_TIMING_SUMMARY=1
VLLM_XPU_DECODE_TIMING_PRINT_EVERY=0
```

The first summary run was BF16 p512/n64 with synchronized rank-0 timing:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n64-20260509T231814Z.log
```

It is diagnostic only because synchronization slows throughput, but the split
is useful:

| label | count | total ms | avg ms |
| --- | ---: | ---: | ---: |
| `runner.forward` | 64 | 3648.739 | 57.012 |
| `moe.quant_apply` | 4154 | 1798.944 | 0.433 |
| `moe.fused_experts_fallback` | 186 | 814.829 | 4.381 |
| `moe.llm_scaler_u4_bridge` | 3968 | 600.819 | 0.151 |
| `moe.router_select` | 4154 | 284.095 | 0.068 |
| `runner.preprocess` | 66 | 76.845 | 1.164 |
| `runner.postprocess` | 64 | 45.556 | 0.712 |

The summary includes prefill and first-token outliers. The steady printed
decode samples were more stable:

```text
runner.forward: about 45 ms/token
moe.router_select: about 0.06 ms/layer
moe.quant_apply: about 0.18 ms/layer
moe.llm_scaler_u4_bridge: about 0.10 ms/layer
```

Since `moe.quant_apply` contains the bridge timing, the useful estimate is
router plus quant/apply: about `0.24 ms/layer`, or about `15 ms/token` across
62 layers under synchronized timing. That is now a large minority rather than
the full decode cost. The remaining ceiling is outside the custom MoE bridge:
attention/KV, Q/K RMS plus TP collectives, projections, and compiled graph
boundaries.

I also fixed the helper so `VLLM_XPU_DECODE_TIMING_PRINT_EVERY=0` now means
summary-only instead of printing every call. Patch artifact:

```text
patches/vllm-xpu-decode-timing-summary-helper-20260509.patch
```

An eager-only timing pass exposes the Python-level model boundaries that torch.compile hides:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n4-20260509T144519Z.log
steady p50 per-layer rank-0 timings:
qkv ~0.034 ms
qk_norm ~0.247 ms
rope ~0.018 ms
kv_attention ~0.098 ms
o_proj ~0.092 ms
router_linear ~0.073 ms
experts_total ~0.145 ms
tp.all_reduce.direct ~0.047 ms, about three calls per layer
```

Eager mode changes the execution and is much slower, so the absolute numbers are not a compiled-path profile. The useful signal is the shape of the work: MiniMax Q/K norm plus repeated TP collectives is a credible XPU fusion target. This matches the existing CUDA-only `minimax_allreduce_rms_qk` path in the vLLM tree, but that op is absent from the XPU build.

An exact-math Q/K contiguous-slice experiment was then tested:

```text
VLLM_MINIMAX_QK_CONTIG=1
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T145504Z.log
Throughput: 0.13 requests/s, 101.77 total tokens/s, 33.92 output tokens/s
```

This is slower than the default-IPC p512/n256 baseline of `34.578045` output tok/s, so forcing contiguous Q/K copies is not the right fix. Keep it only as a negative reproducibility switch in `patches/vllm-xpu-decode-timing-and-qk-contig-20260509.patch`.

## Next Work

The next useful optimization path is to reduce the remaining decode overhead around the same MiniMax MoE path:

- move more route/gather/top-k handling into the custom op so Python/vLLM glue does less per layer;
- keep the BF16-capable path available for quality-preserving runs; speed work should now target overhead outside the MoE matvec itself;
- prototype an XPU equivalent of the CUDA-only MiniMax `minimax_allreduce_rms_qk` fusion, or at least fuse Q/K variance, small allreduce, and RMS scaling more tightly than the current PyTorch graph;
- inspect TP4 allreduce/attention decode cost now that MoE is less dominant;
- add cleaner per-rank/per-process timers around the custom MoE bridge and vLLM attention call sites, because the first kernel-only trace shows matvec wait is only part of the decode step;
- revisit XPU graph only if vLLM adds communication-op capture support or if we test a non-TP decode path;
- consider a larger-batch version only if it does not pull prompt/prefill back onto a tiny-M kernel.
