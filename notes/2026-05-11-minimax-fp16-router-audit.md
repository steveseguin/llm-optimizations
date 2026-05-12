# MiniMax FP16 Router Audit, 2026-05-11

## Purpose

Check whether the faster MiniMax M2.7 AutoRound FP16 router projection can be
treated as quality-preserving. The warmed p512/n1536 speed screen reached
`38.13` output tok/s, but it changed the router matmul from FP32 to FP16 before
expert selection.

## Patch

Added a default-off vLLM MiniMax audit:

- `VLLM_MINIMAX_M2_FP16_ROUTER=1` enables the experimental FP16 router.
- `VLLM_MINIMAX_M2_FP16_ROUTER_AUDIT=1` initializes a shadow FP16 gate buffer
  and compares FP32 and FP16 expert routing.
- The final audit compares the same biased MiniMax routing decision used by
  vLLM: `sigmoid(router_logits) + e_score_correction_bias`.
- It logs ordered top-8 mismatches, unordered top-8 set mismatches, and whether
  the FP16 top-16/top-32 candidate set contains the exact biased FP32 top-8.

Patch artifact:

`patches/vllm-minimax-fp16-router-audit-20260511.patch`

Both source and installed venv files passed `py_compile`.

## Final Audit Run

Run shape:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM/XPU 0.20.1-local, TP4, `dtype=float16`
- active route: FP32 router
- shadow route: FP16 router
- `--enforce-eager`
- p64/n16, batch/concurrency 1
- `max_model_len=512`, `max_num_batched_tokens=128`

Log:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-fp16-router-audit-20260511/vllm-minimax-m27-autoround-tp4-p64n16-20260512T031110Z.log`

Aggregate:

```text
audit_lines=496
prefill_lines=248
decode_lines=248
ordered_mismatch_sum=640
set_mismatch_sum=128
top16_candidate_miss_sum=0
top32_candidate_miss_sum=0
max_abs_logit_delta=0.00762177
max_abs_choice_delta=0.000101447
```

The set mismatch is real: layer `model.layers.41.mlp` had `32` prefill tokens
per rank where the unordered FP16 top-8 expert set differed from the exact
biased FP32 top-8. Direct FP16 routing is therefore not promoted.

The candidate coverage result is useful: FP16 top-16 contained the exact biased
FP32 top-8 for every audited token and layer in this smoke. That suggests a
quality-preserving router optimization may be possible by using FP16 only to
generate a small candidate set, then computing exact FP32 logits/scores for
those candidates and feeding exact top-k ids/weights into the MoE apply path.

## Decision

- Keep direct `VLLM_MINIMAX_M2_FP16_ROUTER=1` off for quality-conservative
  benchmarks and LocalMaxxing submissions.
- Do not submit the `38.13` output tok/s FP16-router speed screen, because route
  sets can change.
- Treat candidate-repair routing as a promising next implementation path, but
  do not mask full `router_logits` as a shortcut: MiniMax's expert correction
  bias is applied after sigmoid, so an exact candidate path needs to bypass or
  extend router selection and pass exact `topk_weights`/`topk_ids` to the MoE
  quant method.

Structured data:

`data/minimax-m27-fp16-router-audit-20260511.json`
