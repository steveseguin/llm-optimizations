# 2026-05-06 - FP8 PP2 GDN fallback addendum

## Current State

The Qwen3.6 27B FP8 2x2 topology is still experimental. PP handoff is no longer the main unknown: trace logs show the last PP rank can propose CPU n-gram draft tokens and PP0 can receive scheduled speculative work.

The concrete blocker found today is the XPU GDN custom op. `_gdn_attention_core_xpu_impl()` explicitly asserted that `GDNAttentionMetadata.spec_sequence_masks` must be `None`, so the first real PP2 speculative step failed before GDN recurrent attention could run.

## Patch Direction

Added a source fallback for speculative GDN on XPU:

- reconstruct `mixed_qkv`, `z`, `b`, and `a` from projected XPU tensors;
- copy reconstructed `z` into the custom-op output tensor;
- call `self._forward_core(...)` for speculative GDN metadata;
- keep native `_xpu_C.gdn_attention` for non-speculative GDN.

Also added deterministic `vllm bench latency` prompt controls for repeatable n-gram experiments:

- `VLLM_BENCH_LATENCY_PROMPT_SEED`
- `VLLM_BENCH_LATENCY_PROMPT_MODE=repeat`
- `VLLM_BENCH_LATENCY_REPEAT_PERIOD`

## Validation Status

- Pre-fallback forced-repeat PP2 reproduced the GDN assertion after scheduling 4 draft tokens on PP0.
- Eager mode completed but did not schedule draft tokens in that sample, so it was not a true validation.
- Patched non-eager retry failed before inference during model load with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, followed by `UR_RESULT_ERROR_DEVICE_LOST`.
- Post-failure health checks passed:
  - all four B70s still enumerated through `sycl-ls`;
  - small Torch XPU allocations succeeded on all four devices;
  - llama.cpp Q4_0 3-card sanity reached `45.104294 tok/s` on `p512/n128/r1`.

## Next Work

- Do not submit these PP2 diagnostic runs to LocalMaxxing.
- Give vLLM/XPU a clean runtime window before retrying the fallback validation.
- If the fallback reaches inference, compare tiny deterministic speculative and non-speculative outputs before attempting speed numbers.
- Keep Q4_0 GGUF and TP4 static FP8 as the active speed paths while PP2 GDN speculation remains under validation.

Artifacts:

- Note: `notes/2026-05-06-vllm-pp2-gdn-spec-fallback.md`
- Data: `data/qwen36-fp8-pp2-gdn-spec-fallback-20260506.json`
- Patch: `patches/vllm-xpu-gdn-spec-fallback-and-deterministic-bench-20260506.patch`
