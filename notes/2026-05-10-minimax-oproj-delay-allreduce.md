# MiniMax Delayed Output-Projection Allreduce, 2026-05-10

## Goal

Test whether moving the attention output-projection TP allreduce out of
`RowParallelLinear.forward()` and into the decoder layer immediately before the
existing residual-add RMSNorm improves the compiled schedule.

This is a quality-preserving scheduling experiment. It does not remove or skip
communication:

1. `o_proj` computes local TP partial output with `reduce_results=False`.
2. `MiniMaxM2DecoderLayer.forward()` runs the same
   `tensor_model_parallel_all_reduce(hidden_states)`.
3. The existing `post_attention_layernorm(hidden_states, residual)` handles
   residual add plus RMSNorm exactly as before.

Flag:

```bash
VLLM_MINIMAX_O_PROJ_DELAY_ALLREDUCE=1
```

Runtime:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70
- FP16 activations
- llm-scaler raw-u4 decode-only MoE path enabled
- `max_model_len=2048`, `max_num_batched_tokens=1024`
- XPU graph disabled
- no speculative decoding
- no GPU power-limit changes

## Results

The delayed-output-allreduce runs used:

```text
/mnt/fast-ai/vllm-cache/minimax-oproj-delay-20260510
```

| Run | Shape | Cache state | AOT hash | GPU KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T123405Z` | p512/n512 | fresh compile | `0f118655...` | 9,408 | 57.610 | 28.805 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oproj-delay/vllm-minimax-m27-autoround-tp4-p512n512-20260510T123405Z.log` |
| `20260510T123713Z` | p512/n512 | warmed AOT reload | `0f118655...` | 17,216 | 73.080 | 36.540 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oproj-delay/vllm-minimax-m27-autoround-tp4-p512n512-20260510T123713Z.log` |
| `20260510T123959Z` | p512/n1536 | warmed AOT reload | `0f118655...` | 16,448 | 48.599 | 36.449 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oproj-delay/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T123959Z.log` |

Immediate same-shape p512/n512 baseline before this experiment:

- `35.820` output tok/s, `71.640` total tok/s, `17,216` KV tokens.

Current quality-conservative p512/n1536 reference:

- `37.552538` output tok/s, `50.070051` total tok/s, with Q/K TP variance
  allreduce visible.

## Interpretation

The p512/n512 warm run shows a small local improvement over the immediate
baseline (`36.540` vs `35.820` output tok/s). That suggests the output-projection
allreduce boundary is worth revisiting.

The longer p512/n1536 run is the more stable throughput comparison, and it
regressed to `36.449` output tok/s. It also reported only `16,448` GPU KV-cache
tokens versus the usual `17,216` for the same 2048-window setup.

Decision:

- do not keep `VLLM_MINIMAX_O_PROJ_DELAY_ALLREDUCE` in the active runtime;
- do not submit these runs to LocalMaxxing;
- preserve the patch as a scheduling clue;
- future work should fuse the post-projection allreduce with residual/RMSNorm
  or output-projection epilogue work, not merely move the same allreduce call.
