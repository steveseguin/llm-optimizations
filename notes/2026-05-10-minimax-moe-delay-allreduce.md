# MiniMax Delayed MoE Allreduce Screen, 2026-05-10

## Goal

Test whether moving MiniMax M2.7's late MoE TP allreduce out of the generic MoE
runner and into the MiniMax decoder layer after `block_sparse_moe` changes the
compiled schedule enough to improve throughput.

This is intended as a quality-preserving scheduling experiment for the TP4,
non-EP path:

1. `MoERunner._maybe_reduce_final_output()` skips its late
   `tensor_model_parallel_all_reduce(states)` only when
   `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1`.
2. `MiniMaxM2DecoderLayer.forward()` then runs the same
   `tensor_model_parallel_all_reduce(hidden_states)` immediately after
   `self.block_sparse_moe(hidden_states)`.

It does not drop experts, change routing, change sampling, or raise power
limits.

## Results

The runs used:

```text
VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache/minimax-moe-delay-20260510
```

Runtime:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70
- FP16 activations
- llm-scaler raw-u4 decode-only MoE path enabled
- `max_model_len=2048`, `max_num_batched_tokens=1024`
- XPU graph disabled

| Run | Shape | Cache state | AOT hash | GPU KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T124819Z` | p512/n512 | fresh compile | `f596ee22...` | 9,408 | 56.256 | 28.128 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-moe-delay/vllm-minimax-m27-autoround-tp4-p512n512-20260510T124819Z.log` |
| `20260510T125125Z` | p512/n512 | warmed AOT reload | `f596ee22...` | 17,216 | 73.242 | 36.621 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-moe-delay/vllm-minimax-m27-autoround-tp4-p512n512-20260510T125125Z.log` |
| `20260510T125354Z` | p512/n1536 | warmed AOT reload | `f596ee22...` | 17,216 | 50.288 | 37.716 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-moe-delay/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T125354Z.log` |
| `20260510T125656Z` | p512/n1536 | warmed AOT reload repeat | `f596ee22...` | 17,216 | 49.362 | 37.021 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-moe-delay/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T125656Z.log` |

References:

- Same-shape p512/n512 baseline immediately before these boundary screens:
  `35.820` output tok/s.
- Current quality-conservative p512/n1536 reference with Q/K TP variance
  allreduce visible: `37.552538` output tok/s / `50.070051` total tok/s.

## Interpretation

The p512/n512 warm run again shows a small same-shape improvement over the
immediate baseline. One p512/n1536 run also beat the quality-conservative
reference by a small margin (`37.716` vs `37.552538` output tok/s), but the
repeat fell to `37.021`.

That makes this unpromoted rather than a new speed recipe. It is useful evidence
that the late MoE allreduce boundary can influence scheduling, but just moving
the allreduce is not a reliable improvement. A real optimization probably needs
to fuse the MoE final allreduce with the next residual/RMSNorm boundary or with
the MoE output epilogue.

Decision:

- do not submit these runs to LocalMaxxing;
- keep `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE` unset in the active runtime;
- archive the patch as `patches/vllm-minimax-moe-delay-allreduce-20260510.patch`;
- revisit only as part of a real fused MoE-output allreduce plus layernorm path.
