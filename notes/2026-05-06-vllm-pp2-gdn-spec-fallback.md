# 2026-05-06 - vLLM FP8 PP2 GDN speculative fallback

## Context

Target: Qwen3.6 27B FP8, static compressed-tensors checkpoint, vLLM/XPU on 4x Intel Arc Pro B70.

Primary topology under investigation:

- `TP=2`
- `PP=2`
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`
- 512 prompt tokens
- 32 output tokens for short fault isolation
- CPU n-gram speculative decoding with 4 draft tokens

Wrapper:

`/home/steve/bench-vllm-qwen36-fp8.sh`

Important wrapper change:

- The wrapper now sets `PYTHONPATH=/home/steve/src/vllm` when that source tree exists. This avoids the previous source/venv file mismatch from copying individual vLLM files into site-packages.

## What Changed

Patch:

`/home/steve/llm-optimization-artifacts/patches/vllm-xpu-gdn-spec-fallback-and-deterministic-bench-20260506.patch`

SHA256:

`c8e01513d26b090e45065db83f16107f984e0e47ad41ea645c1a467f5fdde8b4`

Patch contents:

- Added deterministic latency benchmark prompt controls:
  - `VLLM_BENCH_LATENCY_PROMPT_SEED=<int>`
  - `VLLM_BENCH_LATENCY_PROMPT_MODE=repeat`
  - `VLLM_BENCH_LATENCY_REPEAT_PERIOD=<int>`
- Added PP n-gram trace logging behind `VLLM_XPU_TRACE_NGRAM_PP=1`.
- Added a generic GDN speculative branch in `GatedDeltaNetAttention.forward_xpu()`.
- Added a fallback inside `_gdn_attention_core_xpu_impl()`:
  - when `GDNAttentionMetadata.spec_sequence_masks` is present, reconstruct `mixed_qkv`, `z`, `b`, and `a` from the projected XPU tensors;
  - copy the reconstructed `z` into the custom-op output tensor;
  - call `self._forward_core(...)` instead of asserting.
- Kept the native `_xpu_C.gdn_attention` path for non-speculative GDN.

## Findings

The key blocker is now specific:

- PP handoff is working. The last PP rank can propose draft tokens and `take_draft_token_ids()` returns them.
- The first PP stage then receives a scheduled speculative step.
- Before the fallback patch, that step hit:

`/home/steve/src/vllm/vllm/_xpu_ops.py:118`

with:

`assert attn_metadata.spec_sequence_masks is None`

This was an explicit "XPU does not support speculative decoding yet" assertion in the GDN custom op wrapper, not a scheduler ownership issue.

## Runs

### Baseline forced-repeat PP2 failure

Log:

`/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T130708Z.log`

Result:

- The forced-repeat prompt produced `draft_lens=[4]` on the last PP rank.
- PP0 then scheduled `spec_lens={'0-a172e5cc': 4}`.
- The run failed at the GDN XPU custom-op assertion above.

### Eager/generic check

Log:

`/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T131446Z.log`

JSON:

`/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T131446Z.json`

Result:

- Completed with `--enforce-eager`.
- It did not schedule draft tokens in this sample, so it did not exercise the failing GDN speculative step.
- Measured speed was only `7.469 tok/s` for 32 output tokens and is not a useful benchmark.

### Patched non-eager retry

Log:

`/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T131803Z.log`

Result:

- Failed during model weight load before reaching inference.
- Level Zero returned `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, then `UR_RESULT_ERROR_DEVICE_LOST` during cache cleanup.
- `sycl-ls` still enumerated all four B70s afterward, and a small Torch XPU allocation succeeded on all four devices.
- A post-failure llama.cpp Q4_0 3-card sanity run also completed:
  - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-post-vllm-device-lost-sanity-triple213-p512n128-r1-20260506T132352Z.jsonl`
  - Prompt: `116.035680 tok/s`
  - Decode: `45.104294 tok/s`
- Treat this as a runtime stability failure after repeated vLLM load/unload cycles, not a model benchmark and not validation of the fallback patch.

## Interpretation

The fallback patch is source-correct and compiles, but it still needs a clean validation run after the Level Zero memory/device-lost episode.

Do not submit these PP2 runs to LocalMaxxing:

- the assertion failure is diagnostic;
- the eager run did not actually speculate;
- the patched retry failed at load time.

The currently shareable FP8 result remains the already submitted TP4 static FP8 run:

- 512 prompt / 512 output
- `49.581893 tok/s`
- LocalMaxxing ID `cmotql1v60013qy01016jcs7r`

## Next

- Give vLLM/XPU a cool-down or reboot before another PP2 validation, because repeated FP8 loads can leave the Level Zero runtime unstable.
- Re-run the forced-repeat PP2 test with the fallback patch and confirm whether it reaches the generic GDN spec path.
- If the fallback reaches inference, validate correctness against non-spec output on a tiny deterministic prompt before attempting speed runs.
- Keep Q4_0 GGUF and TP4 FP8 as the active speed paths while PP2 GDN speculation remains experimental.
