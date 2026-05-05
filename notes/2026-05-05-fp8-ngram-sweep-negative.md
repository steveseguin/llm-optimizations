# Qwen3.6 FP8 N-Gram Sweep Negatives

Date: 2026-05-05

## Summary

I tested three adjacent n-gram speculative decode settings around the current static FP8 TP4 best for `vrfai/Qwen3.6-27B-FP8`.

None beat the validated best of `47.674832 tok/s` with n-gram `num_speculative_tokens=4`, lookup min/max `2/4`, default IPC/topology recognition, and `CCL_ATL_TRANSPORT=ofi`.

## Common Setup

- Engine: patched vLLM/XPU `0.20.1`;
- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- quantization: compressed-tensors FP8;
- attention: XPU FlashAttention2;
- GPUs: 4x Intel Arc Pro B70;
- selector: `level_zero:0,1,2,3`;
- TP/PP: `4/1`;
- prompt/output: `512/512`;
- measured iterations: `3`, after `1` warmup;
- CCL: `CCL_ATL_TRANSPORT=ofi`, default IPC/topology.

## Results

| Speculative config | Avg latency | Output tok/s | JSON |
| --- | ---: | ---: | --- |
| n-gram tokens `3`, lookup `2/4` | `12.580774859331237` | `40.697016` | `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T042012Z.json` |
| n-gram tokens `4`, lookup `2/3` | `11.870841589993992` | `43.130893` | `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T042304Z.json` |
| n-gram tokens `5`, lookup `2/4` | `11.593161030997484` | `44.163969` | `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T042552Z.json` |

## Decision

Keep the current FP8 recommendation at n-gram tokens `4`, lookup `2/4`.

These runs were not submitted to LocalMaxxing because they are negative boundary runs rather than improved or representative best results.
