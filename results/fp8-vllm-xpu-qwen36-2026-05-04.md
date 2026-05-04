# Qwen3.6 27B FP8 on Intel Arc Pro B70 via vLLM/XPU

Date: 2026-05-04

## Summary

The official `Qwen/Qwen3.6-27B-FP8` model was downloaded and tested on 1/2/4 Intel Arc Pro B70 GPUs using vLLM/XPU. The model is Hugging Face Safetensors with 128x128 block-scaled FP8 weights, not GGUF.

Current vLLM/XPU does not provide a native XPU kernel for this block-FP8 quantization. A local patch adds two fallback paths:

- `XPUBF16Fp8BlockScaledMMLinearKernel`: dequantize block-FP8 weights to BF16 after load and use BF16 matmul. This preserves the checkpoint values after dequantization, but is slow.
- `XPURequantFp8BlockScaledMMLinearKernel`: opt-in with `VLLM_XPU_BLOCK_FP8_REQUANT=1`; dequantize block-FP8 weights, requantize to per-channel FP8, then use Intel `fp8_gemm_w8a16`. This is experimental and may affect quality.

Conclusion: this FP8 path is runnable but not performance-competitive yet. Continue Q4_0 GGUF/SYCL optimization as the primary quality-preserving speed track.

## Environment

- Host: Ubuntu 24.04 LTS
- GPUs: 4x Intel Arc Pro B70 32GB
- Intel compute-runtime: `26.14.37833.4`
- vLLM env: `/home/steve/.venvs/vllm-xpu-managed`
- vLLM: `0.20.1`
- PyTorch: `2.11.0+xpu`
- transformers: `5.7.0`
- vllm-xpu-kernels: `0.1.7`
- Model path: `/home/steve/models/qwen3.6-27b-fp8-hf`
- Benchmark wrapper: `tools/bench-vllm-qwen36-fp8.sh`

## Commands

TP4 requant run that completed:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
VLLM_XPU_BLOCK_FP8_REQUANT=1 \
TP=4 INPUT_LEN=512 OUTPUT_LEN=32 BATCH_SIZE=1 MAX_MODEL_LEN=768 \
GPU_MEM_UTIL=0.95 NUM_ITERS=1 WARMUP_ITERS=0 \
EXTRA_ARGS='--kv-cache-memory-bytes 2G --max-num-seqs 1 --max-num-batched-tokens 768 --enforce-eager' \
/home/steve/bench-vllm-qwen36-fp8.sh
```

TP2 requant run that initialized but failed during generation:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
VLLM_XPU_BLOCK_FP8_REQUANT=1 \
TP=2 INPUT_LEN=512 OUTPUT_LEN=64 BATCH_SIZE=1 MAX_MODEL_LEN=768 \
GPU_MEM_UTIL=0.95 NUM_ITERS=1 WARMUP_ITERS=0 \
EXTRA_ARGS='--kv-cache-memory-bytes 2G --max-num-seqs 1 --max-num-batched-tokens 768 --enforce-eager' \
/home/steve/bench-vllm-qwen36-fp8.sh
```

## Results

| Mode | Output Shape | Outcome | Latency | Approx Output tok/s Including Prefill | Notes |
| --- | --- | --- | ---: | ---: | --- |
| TP1 requant | 512 prompt + 32 output | completed | `14.401 s` | `2.22` | Model load used `27.64 GiB`; only `0.54 GiB` KV headroom. |
| TP2 requant fixed KV | 512 prompt + 64 output | failed | n/a | n/a | Initialized, then OOMed in Intel Triton benchmark cache allocation during generation. |
| TP4 BF16 fallback | 512 prompt + 32 output | completed | `9.870 s` | `3.24` | Quality-preserving fallback but slow. |
| TP4 requant fixed KV | 512 prompt + 32 output | completed | `9.733 s` | `3.29` | Per-card model load `7.03 GiB`; still much slower than Q4_0 llama.cpp. |

Relevant local logs:

- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-tp1-in512-out32-bs1-20260504T115551Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-tp2-in512-out64-bs1-20260504T120559Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-tp4-in512-out32-bs1-20260504T114855Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-tp4-in512-out32-bs1-20260504T120957Z.log`

## Interpretation

- The official FP8 model is present and usable for experiments.
- The useful missing piece is a native XPU 128x128 block-FP8 W8A8 GEMM path. `vllm-xpu-kernels 0.1.7` exposes FP8 kernels, including `fp8_gemm_w8a16`, but not the block-FP8 path needed for this model.
- The TP2 OOM is not a simple model-fit issue. It happened after successful fixed-KV initialization and points at Intel Triton/XPU temporary allocation behavior.
- The requant path is not a fair quality-preserving comparison until quality is checked, because it requantizes the official block-FP8 weights.

## Next Work

- Keep this patch as an R&D artifact for native block-FP8 kernel work.
- Try SGLang/KTransformers later only if their B70/XPU block-FP8 support is better than vLLM's current path.
- Keep the main optimization effort on Qwen3.6 27B Q4_0 GGUF in llama.cpp/SYCL, where the current best quality-preserving result is far ahead of this FP8 path.
