# vLLM PP2 x TP2 N-Gram Drafter Fix

Date: 2026-05-05

## Summary

The PP2 x TP2 static FP8 n-gram speculative path previously crashed because non-last pipeline-parallel ranks did not have `self.drafter`, but `_build_attention_metadata()` dereferenced it whenever speculative decoding was enabled.

I patched the metadata path to use `getattr(self, "drafter", None)`.

## Patch

Files patched locally:

- `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py`;
- `/home/steve/.venvs/vllm-xpu-managed/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`.

Behavior:

- Eagle/DFlash drafters still use their `kv_cache_gid` branch when present;
- n-gram and non-drafter PP ranks use the common metadata path.

## Smoke Result

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- engine: patched vLLM/XPU `0.20.1`;
- topology: PP2 x TP2 across 4x B70;
- quantization: compressed-tensors FP8;
- speculative decode: n-gram, `num_speculative_tokens=4`, lookup min/max `2/4`;
- prompt/output: `32/8`;
- measured iterations: `1` after `1` warmup.

Result:

- completed successfully;
- avg latency: `0.32795706900651567 s`;
- output throughput: `24.393437 tok/s`;
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in32-out8-bs1-20260505T043249Z.json`;
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in32-out8-bs1-20260505T043249Z.log`.

## Larger Run

PP2 x TP2, 512 prompt / 256 output, same n-gram config:

- failed during model load/memory accounting with `UR_RESULT_ERROR_DEVICE_LOST`;
- oneCCL workers then reported broken-pipe cleanup errors after the device-lost failure;
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out256-bs1-20260505T043606Z.log`.

## Decision

The drafter bug is unblocked, but PP2 x TP2 is still not a validated performance path. Do not submit PP2 x TP2 to LocalMaxxing until a 512-token-class run completes.
