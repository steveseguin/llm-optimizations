# 2026-05-06 - Qwen3.6 27B FP8 TP4 longer context

## Context

Target: Qwen3.6 27B FP8 on vLLM XPU, 4x Arc Pro B70, `TP=4`, `PP=1`, `INPUT_LEN=2048`, `OUTPUT_LEN=512`, `MAX_MODEL_LEN=4096`, batch 1.

Runtime rule: do not source oneAPI `setvars.sh` for vLLM. Keep `/home/steve/.venvs/vllm-xpu-managed/lib` first in `LD_LIBRARY_PATH`.

## Result

- Output throughput: `43.688466 tok/s`
- Total throughput: `218.442329 tok/s`
- Average latency: `11.719340 s`
- LocalMaxxing: `cmottw16x002wqy01jvbluobl`

## Configuration

`MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai`

`QUANTIZATION=compressed-tensors`, `KV_CACHE_DTYPE=auto`, `GPU_MEM_UTIL=0.85`, `CCL_ATL_TRANSPORT=ofi`, `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`.

Speculative config:

`{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":4}`

## Interpretation

- The longer-context run is stable and useful as a 4096-context datapoint.
- Decode is lower than the 512/512 best (`49.581893 tok/s`), mostly because n-gram acceptance drops. The measured benchmark windows after warmup had `28.3%`, `20.1%`, and `18.5%` average draft acceptance.
- Prompt throughput is strong around `204 tok/s` during measured iterations.

## Files

- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in2048-out512-bs1-20260506T085730Z.json`
- Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in2048-out512-bs1-20260506T085730Z.log`
