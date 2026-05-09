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
```

## What Changed

The earlier llm-scaler experiment converted packed INT4 weights to signed compact form and then used the tiny routed MoE path for both prompt and decode shapes. It proved decode speed but regressed full p512 because prefill also hit the tiny path.

The current patch does two narrower things:

- adds an unsigned uint4 variant in `llm-scaler` that decodes each raw nibble as `nibble - 8`;
- gates vLLM so the custom path only runs for MiniMax decode-size batches: `x.dtype == torch.float16` and `x.shape[0] <= 4`.
- accepts FP32 `topk_weight` tensors directly in the custom down-projection kernel so vLLM no longer casts router weights to FP16 for every decode-layer call.

This keeps prompt/prefill on vLLM's normal W4A16 fused-experts path and swaps only the decode MoE work onto the faster ESIMD kernel.

The `512/256` validation confirms the decode-side goal: when the fixed prefill cost is amortized over a longer generation window, single-session output throughput is above 30 tok/s. Removing the router-weight cast improved the same p512/n256 method from `33.033788` to `34.157842` output tok/s.

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

## Next Work

The next useful optimization path is to reduce the remaining decode overhead around the same MiniMax MoE path:

- move more route/gather/top-k handling into the custom op so Python/vLLM glue does less per layer;
- add a BF16-capable variant so the path can run without forcing FP16 activations;
- inspect TP4 allreduce/attention decode cost now that MoE is less dominant;
- revisit XPU graph only if vLLM adds communication-op capture support or if we test a non-TP decode path;
- consider a larger-batch version only if it does not pull prompt/prefill back onto a tiny-M kernel.
