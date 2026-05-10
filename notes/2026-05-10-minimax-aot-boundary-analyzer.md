# MiniMax AOT Boundary Analyzer, 2026-05-10

## Tool

Added `scripts/analyze-vllm-aot-allreduce-boundaries.py`.

Purpose: classify compiled vLLM AOT allreduce boundaries so each experiment can
record whether it changes the communication graph instead of relying on manual
inspection.

Example:

```bash
scripts/analyze-vllm-aot-allreduce-boundaries.py \
  /home/steve/.cache/vllm/torch_compile_cache/4e4b550b71
```

## Latest Clean Graph

Input graph:

`/home/steve/.cache/vllm/torch_compile_cache/4e4b550b71/rank_0_0/backbone/computation_graph.py`

Summary for one rank:

| Boundary class | Shape | Count |
| --- | --- | ---: |
| hidden-state allreduce feeding RMSNorm | `f16[s72, 3072]` | `62` |
| hidden-state allreduce feeding MoE output handling | `f16[s72, 3072]` | `63` |
| Q/K RMS variance allreduce feeding Q/K RMS apply | `f32[s72, 2]` | `62` |

Structured output:

`data/minimax-m27-aot-boundary-analysis-p512n128-latency-20260510.json`

## Interpretation

This confirms the current optimization target more cleanly:

- the `62` Q/K scalar allreduces are correctness-critical but small;
- the `125` hidden-state allreduces are the bigger source of wait/copy/fence
  boundaries;
- the next useful source patch should target a hidden-state allreduce boundary
  feeding RMSNorm or MoE epilogue work, not another standalone RMSNorm provider
  swap.
