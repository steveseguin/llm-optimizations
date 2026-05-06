# 2026-05-06 FP8 MTP block-FP8 clean-load follow-up

## Context

The prior Qwen3.6 FP8 MTP hybrid test proved that the static compressed-tensors model can discover the dynamic `mtp.safetensors` shard, but scale tensors were still skipped for packed MTP projections. That made the run useful as a loader diagnostic, not a clean MTP performance result.

This follow-up isolates the mismatch:

- target weights: `/home/steve/models/qwen3.6-27b-fp8-vrfai`, static compressed-tensors FP8;
- MTP shard: `/home/steve/models/qwen3.6-27b-fp8-hf/mtp.safetensors`, dynamic block-FP8 with `weight_block_size=[128,128]`;
- hybrid dir: `/home/steve/models/qwen3.6-27b-fp8-vrfai-mtp-hybrid`.

## Patches

Patch:

`/home/steve/llm-optimization-artifacts/patches/vllm-qwen35-mtp-force-block-fp8-clean-20260506.patch`

Behavior:

- add opt-in env `VLLM_QWEN35_MTP_FORCE_FP8_BLOCK=1`;
- leave the target model on compressed-tensors FP8;
- create a local block-FP8 `Fp8Config` for Qwen3.5 MTP drafter layers only;
- keep `mtp.fc` BF16 by ignoring it in the MTP quant config;
- register the env var in `vllm/envs.py`;
- preserve the earlier packed-loader fix that compares mapping candidates against the immutable checkpoint tensor name.

Wrapper patch:

`/home/steve/llm-optimization-artifacts/patches/bench-vllm-qwen36-fp8-selector-unset-20260506.patch`

Behavior:

- default `SELECTOR` to empty instead of `level_zero:0`;
- unset `ONEAPI_DEVICE_SELECTOR` unless explicitly provided;
- add `PYTHONPATH=/home/steve/src/vllm` support so the wrapper uses the patched source tree.

This is necessary for TP4. With `ONEAPI_DEVICE_SELECTOR=level_zero:0`, vLLM/XPU workers see only one visible XPU and TP4 can fail with device index errors.

## Validation

Both corrected runs used TP4/PP1 on four B70s, static compressed-tensors target weights, dynamic block-FP8 MTP shard, and:

```bash
VLLM_QWEN35_MTP_FORCE_FP8_BLOCK=1
VLLM_XPU_BLOCK_FP8_REQUANT=1
MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai-mtp-hybrid
QUANTIZATION=compressed-tensors
TP=4
PP=1
```

The logs selected:

`XPURequantFp8BlockScaledMMLinearKernel for Fp8LinearMethod`

The corrected logs no longer show missing packed `weight_scale_inv` tensors, and the bogus names `qkqkv_proj` and `gate_gate_up_proj` are gone.

## Results

| Run | Prompt | Output | Mode | Avg latency | Output tok/s |
| --- | ---: | ---: | --- | ---: | ---: |
| `20260506T171546Z` | 32 | 8 | eager MTP | `3.383753568 s` | `2.364238` |
| `20260506T171822Z` | 32 | 32 | compiled/async MTP | `17.346884512 s` | `1.844712` |

Logs:

- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in32-out8-bs1-20260506T171546Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in32-out32-bs1-20260506T171822Z.log`

JSON:

- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in32-out8-bs1-20260506T171546Z.json`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in32-out32-bs1-20260506T171822Z.json`

## Decision

Do not submit these runs to LocalMaxxing. The loader is clean now, but MTP decode is dramatically slower than the validated non-spec static FP8 TP4 baseline and the validated CPU n-gram speculative result.

Next MTP work should focus on scheduler/speculative implementation cost, acceptance behavior, and whether Qwen3.6's MTP head is being driven in the intended fast path on XPU. The immediate production-quality paths remain static FP8 TP4 with CPU n-gram speculative decoding and Q4_0 GGUF TP3.
