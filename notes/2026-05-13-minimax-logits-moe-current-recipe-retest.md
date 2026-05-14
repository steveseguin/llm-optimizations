# MiniMax Logits MoE Current-Recipe Retest

Date: 2026-05-13

I retested the exact MiniMax logits MoE path after the current best graph
recipe was found:

- TP4, FP16 activations
- XPU graph enabled with graph partitioning
- attention delayed allreduce enabled
- `--block-size 256`
- `MAX_BATCHED_TOKENS=512`
- `--no-enable-prefix-caching`
- llm-scaler INT4 MiniMax logits decode path enabled

The goal was quality-preserving: move MiniMax router/top-k work into the
llm-scaler INT4 MoE call without changing weights, routing semantics, KV dtype,
sampler, speculative decode, or power limits.

## Result

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| warmup/AOT | 512/128 | `84.668830` | not reported |
| MiniMax logits MoE retest | 512/1536 | `97.619374` | `73.214530` |

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T235204Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T235723Z.log`
- cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-minimaxlogits-20260513T235204Z`
- AOT hash: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

The log confirmed the intended path:

```text
Using llm-scaler XPU INT4 MiniMax logits decode path
```

## Decision

Do not promote or submit. This is a quality-preserving path, but it is slightly
below the validated current best (`73.306312` output tok/s submitted to
LocalMaxxing, repeat mean `73.244155` output tok/s). Keep
`VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS` unset for the recommended recipe.

Useful narrowing: the exact logits path is no longer worth treating as an open
performance lead for the current graph/block/MBT/no-prefix recipe. The next
speed work should stay focused on graph scheduling, collective placement, and
lower-level MoE/projection epilogues.
