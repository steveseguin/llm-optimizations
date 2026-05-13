# MiniMax Current-Best AOT Collective Census, 2026-05-13

## Context

I re-ran the AOT collective classifier on the current MiniMax best cache:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-20260513T171301Z
```

This is the cache used by the `73.306312` output tok/s repeat high, not the
older pre-breakthrough cache. AOT hash:

```text
d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c
```

## Tool Update

`scripts/classify-vllm-aot-collectives.py` now refines missing source/shape
cases using producer and consumer context. This fixes attention hidden
collectives whose generated buffers lack shape metadata but are clearly:

```text
unified_attention_with_output -> int4_gemm_w4a16 -> all_reduce -> RMSNorm
```

## Current Graph Census

Across the eight generated Inductor files:

- `1496` actual `_c10d_functional.all_reduce_` calls.
- `1496` matching `wait_tensor` calls.
- Every collective is fenced two generated-code lines after launch.
- Each generated decode graph has `187` collectives:
  - `1` embedding hidden allreduce.
  - `62` Q/K RMS variance allreduces.
  - `62` attention output hidden allreduces.
  - `62` MoE output hidden allreduces.

## Implication

The remaining speed target is not another scheduler flag. The current graph
immediately waits on every collective, so the quality-preserving work needs to
reduce launch/fence boundaries or fuse useful epilogue work around them:

1. Hidden allreduce plus residual/RMSNorm after attention output.
2. Hidden allreduce plus residual/RMSNorm or next projection around MoE output.
3. Exact Q/K variance allreduce plus RMS apply only if the target math stays
   identical.

Data summary:

```text
data/minimax-m27-current-best-aot-collectives-20260513.json
```
