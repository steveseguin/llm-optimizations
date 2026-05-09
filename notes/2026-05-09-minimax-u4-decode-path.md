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
| unsigned llm-scaler decode-only + default CCL IPC + `MAX_MODEL_LEN=4096` | 512/512 | 29.787984 | 59.575969 | negative compile/KV-profile comparison |
| unsigned llm-scaler decode-only + default CCL IPC + async engine | 512/512 | 36.807084 | 73.614167 | neutral/slightly slower |
| unsigned llm-scaler decode-only + default CCL IPC + detokenize disabled | 512/512 | 37.124066 | 74.248133 | neutral; detokenization is not the bottleneck |

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

Standalone XCCL allreduce probes with explicit `CCL_ZE_IPC_EXCHANGE=pidfd` are fast at the MiniMax hidden-state payload sizes:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/xccl-decode-size-allreduce-4xb70-pidfd-20260509T124512Z.log
minimax_hidden_fp16, 3072 elements, 6144 bytes: 0.015159 ms
minimax_hidden_fp32, 3072 elements, 12288 bytes: 0.015141 ms
small_20kb_fp32, 5120 elements, 20480 bytes: 0.014614 ms
```

This says raw standalone XCCL latency is not large enough by itself to explain a roughly `26.9 ms/token` p512/n512 decode step. The remaining bottleneck is likely embedded graph/scheduler gaps, attention/KV work, router/top-k overhead, or the custom MoE bridge/kernels in context, rather than simple allreduce bandwidth alone. The same standalone allreduce script hung with `CCL_ZE_IPC_EXCHANGE` unset/default, while vLLM default IPC still benchmarks correctly; keep using explicit `pidfd` for standalone communication probes.

## Next Work

The next useful optimization path is to reduce the remaining decode overhead around the same MiniMax MoE path:

- move more route/gather/top-k handling into the custom op so Python/vLLM glue does less per layer;
- add a BF16-capable variant so the path can run without forcing FP16 activations;
- inspect TP4 allreduce/attention decode cost now that MoE is less dominant;
- revisit XPU graph only if vLLM adds communication-op capture support or if we test a non-TP decode path;
- consider a larger-batch version only if it does not pull prompt/prefill back onto a tiny-M kernel.
