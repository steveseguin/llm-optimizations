# MiniMax Generated AOT Analyzer And Env-Constant Screen, 2026-05-11

## Purpose

Two follow-ups after the stock allreduce/RMS pass screen:

1. Update the AOT allreduce-boundary analyzer so it also works on the current
   generated Inductor cache layout, where `computation_graph.py` is no longer
   present.
2. Test whether making the MiniMax default-off helper env switches import-time
   constants improves Dynamo/AOT graph quality.

The performance target remains the accepted quality-cleared MiniMax AutoRound
TP4 result:

- p512/n1536 output: `37.552538` tok/s
- p512/n1536 total: `50.070051` tok/s
- LocalMaxxing: `cmozow03v005wlo01q81bnspx`

## Analyzer Update

Updated:

`scripts/analyze-vllm-aot-allreduce-boundaries.py`

The script now supports both layouts:

- old layout: `rank_0_0/backbone/computation_graph.py`;
- current layout: generated `inductor_cache/**/*.py` files.

For generated Inductor files it scans executable
`torch.ops._c10d_functional.all_reduce_.default(...)` and
`wait_tensor.default(...)` call sites, tracks visible XPU tensor shapes, and
classifies the next boundary as one of:

- `embedding_to_rms_int4_gemm`
- `hidden_to_moe`
- `hidden_to_rms`
- `qk_variance`

Important caveat: the generated-cache count is a representative executable-code
count, not the old full FX graph count. For current MiniMax p512/n1536 controls
it reports `28` representative allreduces rather than the old `187` full graph
allreduce census. It is still useful as a guardrail because it confirms whether
the Q/K variance path and hidden-state allreduce/RMS/MoE categories are present.

Current-control analyzer output:

```json
{
  "layout": "generated_inductor_cache",
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

## Current Control

I reran the current default path at the comparison shape using the existing warm
cache root:

`/mnt/fast-ai/vllm-cache/minimax-current-control-p512n512-20260511`

| Run | Shape | AOT hash | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| current warm | p512/n1536 | `9f17eb...` | 17,216 | `48.924829` | `36.693622` | below reference |

Log:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-current-20260511/vllm-minimax-m27-autoround-tp4-p512n1536-20260511T032123Z.log`

## Env-Constant Screen

Patch tested in active source and installed venv, then reverted:

```python
_MINIMAX_QK_NORM_XPU_DIRECT_ENABLED = (
    os.environ.get("VLLM_MINIMAX_QK_NORM_XPU_DIRECT", "0") == "1"
)
_MINIMAX_AR_FUSED_ADD_RMS_XPU_ENABLED = (
    os.environ.get("VLLM_MINIMAX_AR_FUSED_ADD_RMS_XPU", "0") == "1"
)
```

The intent was to keep opt-in helper behavior while avoiding forward-time
`os.environ.get(...)` calls in graph-shaped code.

| Run | Shape | AOT hash | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| env-constant cold | p512/n512 | `6f26d7...` | 9,408 | `55.486696` | `27.743348` | cold AOT/KV artifact |
| env-constant warm | p512/n512 | `6f26d7...` | 17,216 | `70.954296` | `35.477148` | small short-shape lift |
| env-constant warm | p512/n1536 | `6f26d7...` | 17,216 | `48.600120` | `36.450090` | long-shape regression |

The analyzer reported the same generated collective categories for the
env-constant AOT hash as for the current control:

```json
{
  "f16[s72, 3072] -> embedding_to_rms_int4_gemm": 4,
  "f16[s72, 3072] -> hidden_to_moe": 8,
  "f16[s72, 3072] -> hidden_to_rms": 8,
  "f32[s72, 2] -> qk_variance": 8
}
```

## Decision

Do not keep the env-constant change in the active runtime. It improved p512/n512
slightly but regressed the comparison p512/n1536 shape. No LocalMaxxing
submission.

The useful artifact is the updated analyzer. Future positive-looking runs must
include this generated-cache census so we do not promote a faster graph that
quietly removed the Q/K variance allreduce path.

Structured data:

`data/minimax-m27-generated-aot-analyzer-envconst-20260511.json`
