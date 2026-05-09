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

Key log:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T015634Z.log
Throughput: 0.23 requests/s, 148.74 total tokens/s, 29.75 output tokens/s

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T022204Z.log
Throughput: 0.13 requests/s, 99.10 total tokens/s, 33.03 output tokens/s
```

## What Changed

The earlier llm-scaler experiment converted packed INT4 weights to signed compact form and then used the tiny routed MoE path for both prompt and decode shapes. It proved decode speed but regressed full p512 because prefill also hit the tiny path.

The current patch does two narrower things:

- adds an unsigned uint4 variant in `llm-scaler` that decodes each raw nibble as `nibble - 8`;
- gates vLLM so the custom path only runs for MiniMax decode-size batches: `x.dtype == torch.float16` and `x.shape[0] <= 4`.

This keeps prompt/prefill on vLLM's normal W4A16 fused-experts path and swaps only the decode MoE work onto the faster ESIMD kernel.

The `512/256` validation confirms the decode-side goal: when the fixed prefill cost is amortized over a longer generation window, single-session output throughput is above 30 tok/s.

## Correctness Boundary

This is not speculative decoding and does not drop experts. No sampling parameters, router behavior, KV dtype, or GPU power limits changed.

The model quality boundary is still the selected AutoRound INT4 checkpoint. Relative to the previous AutoRound vLLM baseline, this patch changes only the MoE kernel/dequant path. The exact MiniMax MoE microbench measured max absolute difference around `3.052e-05` versus vLLM fused experts, and the raw-u4 decode path matched the signed-compact compatibility path exactly in the nibble conversion check.

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

## Next Work

The next useful optimization path is to reduce the remaining decode overhead around the same MiniMax MoE path:

- move more route/gather/top-k handling into the custom op so Python/vLLM glue does less per layer;
- add a BF16-capable variant so the path can run without forcing FP16 activations;
- inspect TP4 allreduce/attention decode cost now that MoE is less dominant;
- consider a larger-batch version only if it does not pull prompt/prefill back onto a tiny-M kernel.
