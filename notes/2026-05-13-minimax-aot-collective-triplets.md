# MiniMax M2.7 AOT Collective Triplet Map

Date: 2026-05-13

## Context

The current quality-cleared MiniMax AutoRound INT4 anchor is the TP4
`48.092806751180824` output tok/s run at p512/n1536:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- dtype: `float16`
- TP: `4`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `CCL_TOPO_P2P_ACCESS=1`
- `VLLM_XPU_ENABLE_XPU_GRAPH=0`
- `--async-engine`
- `--compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1]}`
- AOT hash: `3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`

## Classifier Update

`scripts/classify-vllm-aot-collectives.py` now classifies MiniMax AOT allreduce
call sites by source-index triplet instead of reporting most hidden-state
buffers as `unknown`.

Across the eight generated Inductor files in the accepted AOT cache:

- `1496` actual `_c10d_functional.all_reduce_` calls.
- `1496` matching `wait_tensor` calls.
- `1496` allreduce/wait pairs are fenced within seven lines.
- Every matched wait is exactly two generated-code lines after its allreduce.
- Category counts:
  - `496` Q/K RMS variance reductions.
  - `496` attention `o_proj` hidden-state reductions.
  - `496` MoE hidden-state reductions.
  - `8` embedding hidden-state reductions.

Per generated decode graph, this is one embedding reduction plus 62 repeating
triplets:

1. `all_reduce_(3L+1)`: Q/K RMS variance, logically `fp32[tokens, 2]`.
2. `all_reduce_(3L+2)`: attention `o_proj` hidden state, logically
   `fp16[tokens, 3072]`.
3. `all_reduce_(3L+3)`: MoE hidden state, logically `fp16[tokens, 3072]`.

## Optimization Implication

The >60 tok/s path should not remove or approximate the Q/K variance reductions
because that changes model math. The quality-safe target is to reduce launch and
fence overhead around the hidden-state boundaries, or fuse each hidden-state
allreduce with the immediately following residual/RMSNorm/GEMM or MoE epilogue
work.

The current generated code immediately fences every collective. That makes
generic CCL tuning less likely to deliver the full 25% uplift by itself, but
XPU graph capture and narrow fused epilogues are still valid quality-preserving
experiments.

## Data

- `data/minimax-m27-aot-collective-triplets-20260513.json`
