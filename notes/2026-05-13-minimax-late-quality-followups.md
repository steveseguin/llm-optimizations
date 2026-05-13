# MiniMax Late Quality-Preserving Followups, 2026-05-13

## Purpose

Continue trying to beat the MiniMax M2.7 AutoRound 4x B70 TP4 single-session
result without lowering quality. Guardrails stayed unchanged: no model/quant,
router precision, expert routing, KV dtype, sampler, speculative decoding, or
power-limit changes. The baseline remained:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- engine: vLLM `0.20.1-local`, XPU/Level Zero
- hardware: 4x Intel Arc Pro B70 32GB
- recipe: TP4, FP16, llm-scaler INT4 MoE decode, XPU graph, MiniMax attention
  delayed allreduce, `--block-size 256`, `MAX_BATCHED_TOKENS=512`,
  `--no-enable-prefix-caching`, p512/n1536
- AOT hash: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

## Results

| Run | Delta | Output tok/s | Total tok/s | Decision |
| --- | --- | ---: | ---: | --- |
| stream interval | `--stream-interval 16` | `72.860033` | `97.146710` | negative |
| MoE delayed allreduce | `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` | `72.250337` | `96.333783` | negative |
| performance mode | `--performance-mode interactivity` | `72.561494` | `96.748659` | negative |
| performance mode | `--performance-mode throughput` | `72.790178` | `97.053570` | negative |
| balanced repeat | unchanged control, existing AOT | `73.306312` | `97.741749` | new measured high |

The new high is a small repeat improvement over the prior `73.232418` output
tok/s record. It is not a new optimization; it is the same recipe and same AOT
cache landing at the top of the observed run-to-run band.

LocalMaxxing accepted the repeat high as `cmp4f31dh000amz01uqa329e6`.

## Interpretation

The latest scheduler-level knobs did not move the ceiling. Balanced mode remains
best for the batch-1 p512/n1536 shape. Delaying the MoE allreduce in addition
to the attention delayed allreduce also regressed under the current graph path.

This points back to source-level work around the compiled graph boundaries:

1. Hidden-state allreduce plus residual/RMSNorm or projection/MoE epilogue
   fusion.
2. Q/K RMS variance allreduce fusion only if it preserves the target math.
3. Lower-level timing that survives AOT graph replay, because Python timing
   wrappers are bypassed by the warmed graph.

## Artifacts

- Data summary:
  `data/minimax-m27-quality-preserving-followups-20260513-late.json`
- New high log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T184221Z.log`
- New high JSON:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T184221Z.json`
- LocalMaxxing response:
  `data/localmaxxing-responses/minimax-m27-autoround-xpugraph-attn-delay-block256-mbt512-noprefix-repeat-high-20260513.response.json`
