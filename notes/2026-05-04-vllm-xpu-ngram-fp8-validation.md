# vLLM XPU N-Gram FP8 Validation

Date: 2026-05-04

## Summary

N-gram speculative decode is now a valid quality-preserving speed path for the patched TP4 static FP8 vLLM/XPU setup.

Model: `vrfai/Qwen3.6-27B-FP8` at `/home/steve/models/qwen3.6-27b-fp8-vrfai`

Hardware: 4x Intel Arc Pro B70 32GB, Ubuntu 24.04, AMD EPYC 9015 8-Core Processor, 15.2 GB system RAM.

Engine: vLLM `0.20.1`, XPU/FA2, tensor parallel 4, compressed-tensors FP8, auto/BF16 KV, language-model-only.

## Required Local Patches

- Singleton compressed-tensors attention scales reshaped to scalar views for Intel XPU FA2.
- `--language-model-only` skips unused Qwen3.5 vision components.
- XPU Gated DeltaNet speculative metadata contiguity fix.
- Benchmark wrapper accepts `QUANTIZATION=compressed-tensors` and `SPECULATIVE_CONFIG` as a single quoted argument.

## Command Shape

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai \
QUANTIZATION=compressed-tensors \
TP=4 \
INPUT_LEN=512 \
OUTPUT_LEN=512 \
NUM_ITERS=3 \
WARMUP_ITERS=1 \
MAX_MODEL_LEN=1024 \
GPU_MEM_UTIL=0.90 \
SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":5,"prompt_lookup_min":2}' \
EXTRA_ARGS='--disable-log-stats --no-enable-prefix-caching' \
/home/steve/bench-vllm-qwen36-fp8.sh
```

## Results

### 512 Prompt / 256 Output Screen

JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out256-bs1-20260504T220138Z.json`

Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out256-bs1-20260504T220138Z.log`

Average latency: `6.059945654 s`

Computed output throughput: `42.244603 tok/s`

Prior same-shape TP4 FP8 FA2 baseline: about `39.264 tok/s`.

### 512 Prompt / 512 Output Validation

JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T220350Z.json`

Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T220350Z.log`

Latencies: `11.795413791`, `12.598177528`, `11.756794460`

Average latency: `12.050128593 s`

Computed output throughput: `42.489173 tok/s`

Computed total throughput: `84.978346 tok/s`

Prior same-shape TP4 FP8 FA2 baseline: `41.503 tok/s`.

LocalMaxxing submission: `cmorr43b30004jj04h4hhb6v1`, status `APPROVED`.

## Interpretation

The improvement is modest on the longer validation shape, but it is real enough to keep n-gram speculative decode in the optimization track. It preserves the FP8 checkpoint, uses auto/BF16 KV instead of FP8 KV, and does not change GPU power limits.

The LocalMaxxing `backend` field was omitted because the API backend enum currently has no `xpu` value; the run notes and engine flags identify the backend as vLLM XPU/FA2.

Next screens should sweep `num_speculative_tokens` and prompt lookup window on the 512/512 shape before spending more time on MTP.
