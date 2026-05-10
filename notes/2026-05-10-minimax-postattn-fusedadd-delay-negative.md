# MiniMax Post-Attention Fused-Add RMS And Delayed O-Projection Screen, 2026-05-10

## Goal

Close out the simple post-attention residual/RMSNorm optimization branch before
moving to deeper collective-boundary fusion.

Two default-off variants were tested:

1. `VLLM_MINIMAX_POST_ATTN_FUSED_ADD_RMS_XPU=1`
   - Keep the normal attention output-projection allreduce inside
     `RowParallelLinear`.
   - Replace `post_attention_layernorm(hidden_states, residual)` with the XPU
     `fused_add_rms_norm` op.
2. `VLLM_MINIMAX_POST_ATTN_FUSED_ADD_RMS_XPU=1` plus
   `VLLM_MINIMAX_O_PROJ_DELAY_ALLREDUCE=1`
   - Ask `o_proj` to return local TP partials with `reduce_results=False`.
   - Run the same TP allreduce in the decoder layer just before the fused
     residual-add RMSNorm.

Both variants are quality-preserving scheduling/provider experiments. They do
not remove Q/K TP variance allreduce, skip experts, change quantization, enable
speculation, or change GPU power limits.

## Results

All runs:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM/XPU TP4 on 4x Intel Arc Pro B70 32GB
- dtype: FP16 activation path with the llm-scaler raw-u4 decode MoE bridge
- shape: p512/n512, `max_model_len=2048`, `max_num_batched_tokens=512`
- no speculative decoding

| Variant | Run | Cache state | GPU KV tokens | Total tok/s | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| post-attn fused add/RMS only | `20260510T163516Z` | cold isolated AOT | 9,472 | 55.366871 | 27.683436 | cold artifact |
| post-attn fused add/RMS only | `20260510T163832Z` | warm AOT reload | 17,984 | 70.154567 | 35.077284 | negative |
| delayed `o_proj` allreduce + fused add/RMS | `20260510T164227Z` | cold isolated AOT | 9,472 | 52.487691 | 26.243846 | cold artifact |
| delayed `o_proj` allreduce + fused add/RMS | `20260510T164542Z` | warm AOT reload | 17,984 | 71.607175 | 35.803587 | negative |

Reference comparisons:

- accepted p512/n512 MiniMax AutoRound reference: `39.610585` output tok/s
- quality-conservative p512/n1536 anchor: `37.552538` output tok/s /
  `50.070051` total tok/s with Q/K TP variance allreduce enabled

Logs:

```text
/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-postattn-fusedadd/vllm-minimax-m27-autoround-tp4-p512n512-20260510T163516Z.log
/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-postattn-fusedadd/vllm-minimax-m27-autoround-tp4-p512n512-20260510T163832Z.log
/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oproj-delay-fusedadd/vllm-minimax-m27-autoround-tp4-p512n512-20260510T164227Z.log
/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oproj-delay-fusedadd/vllm-minimax-m27-autoround-tp4-p512n512-20260510T164542Z.log
```

Patch artifact:

```text
patches/vllm-minimax-postattn-fusedadd-delay-negative-20260510.patch
```

## Interpretation

The standalone XPU fused-add RMS provider is not enough to beat the installed
runtime reference. Pairing it with delayed output-projection allreduce is also
not enough: the best warm p512/n512 result was only `35.803587` output tok/s,
still well below the accepted `39.610585` reference.

This agrees with the source-tree IR screen: MiniMax needs a real fused
collective boundary, not just an RMSNorm provider swap and not just moving the
same allreduce call.

## Decision

- Do not submit these runs to LocalMaxxing.
- Keep both env flags unset for real benchmarks:
  - `VLLM_MINIMAX_POST_ATTN_FUSED_ADD_RMS_XPU`
  - `VLLM_MINIMAX_O_PROJ_DELAY_ALLREDUCE`
- Preserve the patch as a scheduling clue.
- Next work should target allreduce plus residual/RMSNorm fusion or MoE/output
  epilogue fusion in one graph-safe XPU path.
