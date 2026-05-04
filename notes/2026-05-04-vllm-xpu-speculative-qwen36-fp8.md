# vLLM XPU Speculative Decode Follow-Up

Date: 2026-05-04

## Scope

Model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`

Engine shape: vLLM `0.20.1`, XPU, 4x Intel Arc Pro B70, `--tensor-parallel-size 4`, `--quantization compressed-tensors`, `--language-model-only`, FlashAttention2, auto/BF16 KV.

## Tooling Fix

`/home/steve/bench-vllm-qwen36-fp8.sh` previously forced `--quantization fp8`, which is invalid for the VRFAI static FP8 compressed-tensors checkpoint. It is now controlled by `QUANTIZATION`, with `QUANTIZATION=compressed-tensors` for this checkpoint and `QUANTIZATION=none`/`auto` to omit the flag. The wrapper also prints computed output and total tok/s from `avg_latency`.

## Patches

- `vllm/_xpu_ops.py`: make `non_spec_state_indices_tensor` and `non_spec_query_start_loc` contiguous before passing them into `_xpu_C.gdn_attention`.
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`: when XPU Gated DeltaNet sees speculative sequence masks, route through generic `_forward_core` instead of the fused XPU op that asserts on speculative masks.

Expected quality impact: none. These patches preserve weights and sampling behavior; they only change runtime metadata/control-flow handling for speculative decode.

## Results

### MTP Smoke

Command shape: TP4, `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, 32 prompt / 8 output.

Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-mtp1-smoke-in32-out8-20260504T214352Z.log`

Outcome: target `Qwen3_5ForConditionalGeneration` and draft `Qwen3_5MTP` resolved, XCCL ranks initialized, then the run hung before useful generation. A short `strace` sample showed rank 1 spinning in `sched_yield` with 634,289 calls in 5 seconds. Treat this as a startup/synchronization blocker, not a decode benchmark.

LocalMaxxing: not submitted.

### N-Gram Before Patch

Command shape: TP4, `--speculative-config '{"method":"ngram","num_speculative_tokens":2,"prompt_lookup_max":5,"prompt_lookup_min":2}'`, 32 prompt / 8 output.

Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-ngram2-smoke-in32-out8-20260504T215101Z.log`

Outcome: model loaded and reached first forward, then failed in the XPU GDN custom op with `RuntimeError: non_spec_state_indices_tensor must be contiguous`.

LocalMaxxing: not submitted.

### N-Gram After Patch

Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-ngram2-contigfix-smoke-in32-out8-20260504T215426Z.log`

JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-ngram2-contigfix-smoke-in32-out8-20260504T215426Z.json`

Result: completed successfully, `avg_latency=6.131081808 s` for 32 prompt / 8 output. Computed output throughput is `1.304827 tok/s`, but this is a cold one-iteration correctness smoke and not a valid speed result.

LocalMaxxing: not submitted.

## Interpretation

N-gram speculative decode is now runnable on the patched XPU path and can be screened against the TP4 FP8 baseline. MTP is blocked earlier in startup/synchronization and needs separate investigation. vLLM disables async scheduling for n-gram speculative decode, so it will only be useful if accepted draft tokens offset that scheduler loss.
