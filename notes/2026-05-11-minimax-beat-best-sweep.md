# MiniMax Beat-Best Recovery Sweep, 2026-05-11

## Goal

Recover or beat the accepted MiniMax M2.7 AutoRound TP4 quality-cleared result:

- p512/n1536 output: `37.552538` tok/s
- p512/n1536 total: `50.070051` tok/s
- LocalMaxxing: `cmozow03v005wlo01q81bnspx`
- quality guardrail: Q/K TP variance allreduce visible in the compiled graph.

The older `41.130667` p512/n1536 result remains a scheduling clue, not the
quality bar, because its `c15860...` AOT graph did not visibly contain the
per-layer Q/K RMS variance allreduce.

## Runs

All runs used 4x B70 TP4, vLLM/XPU, FP16 activations, MiniMax AutoRound INT4
W4A16 weights, llm-scaler raw-u4 MoE decode enabled, XPU graph disabled,
`max_model_len=2048`, `max_num_batched_tokens=1024`, `max_num_seqs=1`, no
speculation, no expert dropping, and stock power limits.

| Run | Shape | Change | AOT | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| baseline cold | p512/n1536 | default path | `18d4c2...` | 9,408 | `43.271981` | `32.457173` | cold compile artifact |
| baseline warm | p512/n1536 | default path | `18d4c2...` | 17,216 | `49.036874` | `36.777638` | below reference |
| baseline warm | p512/n1024 | default path | `18d4c2...` | 17,216-class | `54.152911` | `36.101961` | below reference |
| MoE-delay cold | p512/n1536 | `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` | `4d8f4b...` | 9,408 | `43.379870` | `32.534902` | cold compile artifact |
| MoE-delay warm | p512/n1536 | `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` | `4d8f4b...` | 17,216 | `48.741933` | `36.556450` | below reference |

## Graph Checks

The current default warm graph keeps the full MiniMax TP communication pattern:

- `187` allreduces total;
- `125` hidden-state `f16[s72, 3072]` allreduces;
- `62` Q/K RMS variance `f32[s72, 2]` allreduces.

The MoE-delay warm graph also keeps `187` allreduces with the same shape split.
Its classification changed to route more hidden-state reductions directly into
RMS boundaries, but that did not improve throughput after reboot.

## Decision

No LocalMaxxing submission. These are valid engineering results, but all
measured runs are below the accepted quality-cleared MiniMax score.

The next work should be code, not another flag sweep:

1. Hidden-state allreduce plus residual/RMS fusion for the `f16[s72,3072]`
   boundary after attention output projection and MoE output.
2. A graph-safe Q/K allreduce plus RMS fusion only if it preserves the INT4 MoE
   compiled schedule and the Q/K TP variance allreduce semantics.
3. Use p64/n32 only as a smoke screen; promote only p512/n1536 or longer
   quality-cleared results.

Detailed data: `data/minimax-m27-beat-best-sweep-20260511.json`.
