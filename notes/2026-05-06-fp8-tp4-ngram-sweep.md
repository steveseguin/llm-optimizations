# 2026-05-06 - Qwen3.6 27B FP8 TP4 n-gram sweep

## Context

Target: Qwen3.6 27B FP8 on vLLM XPU, 4x Arc Pro B70, `TP=4`, `PP=1`, `INPUT_LEN=512`, `OUTPUT_LEN=512`, batch 1.

Runtime rule: do not source oneAPI `setvars.sh` for vLLM. Keep `/home/steve/.venvs/vllm-xpu-managed/lib` first in `LD_LIBRARY_PATH`; this avoids the XCCL segfault seen with oneAPI library precedence.

## Runs

All runs used:

`MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai`

`QUANTIZATION=compressed-tensors`, `KV_CACHE_DTYPE=auto`, `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=1024`, `CCL_ATL_TRANSPORT=ofi`, `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`.

| Method | Draft tokens | Lookup | Output tok/s | Notes |
| --- | ---: | --- | ---: | --- |
| no spec | 0 | n/a | 10.278739 | Stable 512/128 smoke only; confirms base TP4/XCCL path works. |
| ngram | 4 | 2..4 | 48.198021 | Post-patch confirmation; close to earlier best 49.581893. |
| ngram | 3 | 2..4 | 43.023391 | Worse. |
| ngram | 5 | 2..5 | 48.298516 | Tied with depth 4, not a new best. |
| ngram | 6 | 2..6 | 41.724557 | Worse; low acceptance. |

## Findings

- The first post-patch TP4 n-gram confirmation hung during CCL initialization before all four ranks emitted topology warnings. A retry completed normally, and a no-spec TP4 run also completed, so this looks like a transient CCL init hang rather than a deterministic vLLM patch regression.
- `ngram` depth 4 and 5 are the viable range for this prompt shape. Depth 3 leaves performance on the table; depth 6 drafts too aggressively and acceptance collapses.
- The previous submitted best remains `49.581893 tok/s` with depth 4. The new depth 4/5 reruns are valid but lower, so they were documented rather than submitted to LocalMaxxing.

## Files

- Depth 4 rerun: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T083428Z.json`
- Depth 3: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T083723Z.json`
- Depth 5: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T084006Z.json`
- Depth 6: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T084247Z.json`
- No-spec smoke: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out128-bs1-20260506T082913Z.json`
