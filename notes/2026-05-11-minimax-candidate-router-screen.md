# MiniMax Candidate-Repair Router Screen, 2026-05-11

## Purpose

Follow up on the FP16-router audit. Direct FP16 routing changed expert sets, but
the audit showed FP16 candidate sets could contain the exact biased FP32 top-8.
This tested a quality-preserving route:

1. FP16 gate proposes top-M experts.
2. Exact FP32 scores are computed only for those candidates.
3. Exact top-8 ids/weights are passed into the existing MoE apply path.

## Implementation

Added default-off hooks:

- `VLLM_MINIMAX_M2_CANDIDATE_ROUTER_TOPM=<N>`
- `VLLM_MINIMAX_M2_CANDIDATE_ROUTER_MAX_TOKENS=4`

The hook stashes precomputed `topk_weights/topk_ids` on the existing vLLM
router, so `BaseRouter.select_experts()` still performs capture, EPLB mapping,
and dtype conversion. The MoE runner custom-op schema is unchanged.

Patch artifact:

`patches/vllm-minimax-candidate-router-screen-20260511.patch`

Both source and installed venv files passed `py_compile`.

## Audit

With the exact biased MiniMax decision `sigmoid(router_logits) +
e_score_correction_bias`, direct FP16 still changed expert sets:

```text
audit_lines=496
ordered_mismatch_sum=640
set_mismatch_sum=128
top12_candidate_miss_sum=0
top16_candidate_miss_sum=0
top32_candidate_miss_sum=0
```

So the candidate approach is mathematically plausible for this smoke, but direct
FP16 routing remains disqualified.

Audit log:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-candidate-router-20260511/vllm-minimax-m27-autoround-tp4-p64n16-20260512T034350Z.log`

## p512/n512 Results

All runs used vLLM/XPU TP4, MiniMax M2.7 AutoRound, `dtype=float16`,
`VLLM_XPU_USE_LLM_SCALER_MOE=1`, `XPU_GRAPH=0`, `max_model_len=2048`,
`max_num_batched_tokens=1024`, batch/concurrency 1, no speculative decoding,
and no power-limit changes.

| Variant | Cache state | AOT hash | KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | ---: | ---: | ---: | --- |
| disabled baseline | cold | `8fc2c1...` | 9,408 | 57.73 | 28.87 | `20260512T033034Z.log` |
| disabled baseline | warm | `8fc2c1...` | 17,216 | 71.07 | 35.54 | `20260512T033342Z.log` |
| top-16 candidate, first implementation | warm | `8fc2c1...` | 15,680 | 72.43 | 36.21 | `20260512T032741Z.log` |
| top-16 candidate, skips full FP32 gate | cold | `0f3c2f...` | 7,808 | 57.08 | 28.54 | `20260512T033724Z.log` |
| top-16 candidate, skips full FP32 gate | warm | `0f3c2f...` | 15,680 | 72.53 | 36.27 | `20260512T034031Z.log` |
| top-12 candidate, skips full FP32 gate | cold | `0f3c2f...` | 7,808 | 57.72 | 28.86 | `20260512T034629Z.log` |
| top-12 candidate, skips full FP32 gate | warm | `0f3c2f...` | 15,680 | 72.33 | 36.17 | `20260512T034934Z.log` |

The generated-cache allreduce analyzer found the expected Q/K variance
signature for the candidate AOT:

```json
{
  "allreduceCount": 28,
  "byShape": {
    "f16[s72, 3072]": 20,
    "f32[s72, 2]": 8
  },
  "byClassification": {
    "embedding_to_rms_int4_gemm": 4,
    "hidden_to_moe": 8,
    "hidden_to_rms": 8,
    "qk_variance": 8
  }
}
```

## Decision

The candidate-repair path is functional and appears quality-preserving in the
small audit, but it is not a meaningful speed breakthrough. The best warm
p512/n512 result was only `36.27` output tok/s, about `2.1%` over the same-code
disabled baseline and still below prior accepted MiniMax references.

Do not submit to LocalMaxxing. Keep the patch as a useful implementation sketch.
The likely next useful version is an XPU fused candidate-selection kernel that
does FP16 proposal, exact FP32 candidate scoring, top-k repair, and weight
normalization without Python-level advanced indexing and extra graph overhead.

Structured data:

`data/minimax-m27-candidate-router-screen-20260511.json`
