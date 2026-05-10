# MiniMax Python-Level Allreduce Plus Fused-Add RMS Custom Op, 2026-05-10

## Goal

Test whether hiding post-attention allreduce plus residual-add RMSNorm behind a
single XPU custom op improves vLLM/Dynamo graph scheduling.

This branch differs from the previous delayed-`o_proj` plus fused-add RMS
screen:

- previous screen: graph still saw `tensor_model_parallel_all_reduce(...)`
  followed by `fused_add_rms_norm(...)`;
- this screen: graph sees one custom op,
  `vllm::minimax_post_attn_ar_fused_rms`, whose Python implementation runs
  `dist.all_reduce(...)` and then `_C.fused_add_rms_norm(...)`.

The experiment remains quality-preserving. It still performs the output
projection allreduce before residual/RMSNorm, does not skip Q/K variance
allreduce, does not skip experts, and does not use speculation.

## Implementation Notes

The first smoke placed `direct_register_custom_op(...)` inside the decoder
forward path. That fails during Dynamo tracing because `infer_schema` is marked
as skipped:

```text
Attempted to call function marked as skipped ... torch/_library/infer_schema.py
```

Moving registration to module construction fixes the compile blocker. The fixed
tiny p1/n4 smoke completed.

Patch artifact:

```text
patches/vllm-minimax-postattn-ar-fused-customop-negative-20260510.patch
```

The active runtime was reverted after the screen.

## Results

All benchmark runs use:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- engine: vLLM/XPU TP4
- hardware: 4x Intel Arc Pro B70 32GB
- dtype: FP16 with llm-scaler raw-u4 decode MoE bridge
- no XPU graph capture
- no speculative decoding

| Run | Shape | Cache state | GPU KV tokens | Total tok/s | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T165346Z` | p1/n4 | bad registration | n/a | n/a | n/a | Dynamo registration failure |
| `20260510T165629Z` | p1/n4 | fixed registration smoke | 4,736 | 2.139570 | 1.711656 | liveness only |
| `20260510T165924Z` | p512/n512 | cold isolated AOT | 9,472 | 48.907410 | 24.453705 | cold artifact / negative |
| `20260510T170237Z` | p512/n512 | warm AOT reload | 17,920 | 65.221199 | 32.610599 | negative |

Reference:

- accepted p512/n512 MiniMax AutoRound reference: `39.610585` output tok/s
- quality-conservative p512/n1536 anchor: `37.552538` output tok/s /
  `50.070051` total tok/s

## Decision

Do not pursue Python-level custom op fusion for this boundary. It is slower
than the simpler delayed-allreduce plus fused-add screen and much slower than
the accepted reference.

The next viable direction is a real C++/SYCL/XPU kernel or compiler pass that
fuses useful epilogue work around the collective boundary without adding Python
dispatch or extra tensor clones.
