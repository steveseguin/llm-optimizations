# vLLM XPU N-Gram4 FP8 Validation

Date: 2026-05-04

## Summary

Increasing n-gram speculative decode from 2 to 4 draft tokens produced the best static FP8 result so far for Qwen3.6 27B on the four B70 system.

Model: `vrfai/Qwen3.6-27B-FP8`

Engine: vLLM `0.20.1`, XPU/FA2, tensor parallel 4, compressed-tensors FP8, auto/BF16 KV, language-model-only.

Hardware: 4x Intel Arc Pro B70 32GB, Ubuntu 24.04.

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
SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":5,"prompt_lookup_min":2}' \
EXTRA_ARGS='--disable-log-stats --no-enable-prefix-caching' \
/home/steve/bench-vllm-qwen36-fp8.sh
```

## Result

Screen run, two measured iterations:

- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T221056Z.json`
- Average latency: `10.731264135 s`
- Output throughput: `47.711061 tok/s`

Validation run, three measured iterations:

- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T221317Z.json`
- Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T221317Z.log`
- Latencies: `11.833764661`, `11.314851924`, `10.194309922`
- Average latency: `11.114308836 s`
- Output throughput: `46.066742 tok/s`
- Total throughput: `92.133484 tok/s`

LocalMaxxing submission: `cmorre1hq000fi30421gxpv3j`, status `APPROVED`.

## Interpretation

This is a quality-preserving speed path relative to the static FP8 target model: n-gram draft tokens are verified by the target model, KV remains auto/BF16, and no power limits were changed.

It is ahead of:

- Prior TP4 FP8 FA2 512/512 baseline: `41.503 tok/s`.
- TP4 FP8 n-gram2 512/512 validation: `42.489 tok/s`.
- Q4_0 three-B70 validation: about `41.659 tok/s`.

Next sweep points: `num_speculative_tokens=6` and `8`, then prompt lookup window tuning.
