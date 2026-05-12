# MiniMax C158 Recheck, Router, And KV Screens, 2026-05-11

## Purpose

Revisit the MiniMax M2.7 AutoRound TP4 p512/n1536 path after the generated
AOT analyzer learned to scan the current Inductor cache layout. The working
question was whether the earlier `41.130667` output tok/s run was actually
missing the Q/K RMS variance allreduce, or whether the first audit was a tool
limitation.

## C158 Recheck

Archived AOT cache:

`/mnt/fast-ai/vllm-cache-exp/minimax-c158-archive-20260510T070154Z/torch_compile_cache/torch_aot_compile/c15860ddb8a1077c5ba1a1ae2d0f86552a357eb56772cdbf02828195b5a363ec`

Analyzer result:

```json
{
  "layout": "generated_inductor_cache",
  "allreduceCount": 40,
  "byShape": {
    "f16[s72, 3072]": 32,
    "f32[s72, 2]": 8
  },
  "byClassification": {
    "embedding_to_rms_int4_gemm": 8,
    "hidden_to_moe": 12,
    "hidden_to_rms": 12,
    "qk_variance": 8
  }
}
```

This reclassifies the accepted `41.130667` output tok/s result
(`cmoz8cow60001pd010klrb8g8`) as likely quality-valid by the current analyzer.
It is still not reproducibly recovered in the current runtime, so it remains the
best accepted speed target rather than the active reproducible floor.

## Retests

All runs used:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM/XPU 0.20.1-local, TP4, `dtype=float16`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `XPU_GRAPH=0`
- `max_model_len=2048`
- `max_num_batched_tokens=1024`
- p512/n1536, batch/concurrency 1
- no speculative decoding and no power-limit changes

| Variant | Cache state | AOT hash | KV tokens | Total tok/s | Output tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| hotpath clean after removing no-op timing wrapper | cold | `3d762d...` | 9,408 | 44.99 | 33.74 | negative |
| MoE Python cleanup after removing logits-topk branch | cold | `3d762d...` | 9,408 | 44.89 | 33.67 | negative |
| graph-shaped c158 recovery attempt | cold | `893224...` | 9,408 | 45.05 | 33.78 | negative cold artifact |
| graph-shaped c158 recovery attempt | warm | `893224...` | 17,216 | 50.26 | 37.69 | valid but below `41.13` |
| FP8 KV E5M2 | cold | `8c5754...` | 18,816 | n/a | n/a | failed: XPU FlashAttention rejects `fp8_e5m2` |
| FP8 KV E4M3 | cold | `eb4a9f...` | 18,816 | 44.76 | 33.57 | negative and quality-risky |
| FP8 KV E4M3 | warm | `eb4a9f...` | 34,496 | 49.53 | 37.15 | negative and quality-risky |
| FP16 router projection | cold | `893224...` | 7,808 | 46.09 | 34.57 | negative cold artifact |
| FP16 router projection | warm | `893224...` | 15,680 | 50.85 | 38.13 | speed-positive but quality-risky |

The `893224...`, `eb4a9f...`, and `3d762d...` generated AOT caches all include
the expected `f32[s72, 2]` Q/K variance allreduce category under the updated
analyzer.

## Decisions

- Do not submit the new `37.69` or `38.13` runs to LocalMaxxing. The accepted
  `41.130667` run is already public and is now likely quality-valid by the
  corrected analyzer.
- Keep FP8 KV cache off for MiniMax on this stack. E5M2 is not wired through XPU
  FlashAttention, and E4M3 is slower while carrying an accuracy warning.
- Keep FP16 router off for quality-conservative runs. It improves the warmed
  comparison from `37.69` to `38.13` output tok/s, but it changes expert routing
  precision and needs route-agreement auditing before promotion.
- The next high-value path is not more launch flags. It is a source-level router
  optimization that preserves FP32 routing decisions, or a graph-safe fusion
  around Q/K RMS and hidden-state collective boundaries.

Structured data:

`data/minimax-m27-c158-recheck-router-kv-screens-20260511.json`

Patch artifacts:

- `patches/vllm-minimax-graph-shaped-router-kv-screens-20260511.patch`
- `patches/llm-scaler-minimax-u4-logits-topk-negative-20260511.patch`
