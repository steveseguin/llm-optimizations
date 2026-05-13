# MiniMax AOT Collective Classifier, 2026-05-13

I added `scripts/classify-vllm-aot-collectives.py` to make the generated AOT
graph target measurable. The older shell grep summary was useful, but it could
mix FX comments and actual Python calls. The new classifier records actual
`_c10d_functional.all_reduce_` calls, matching `wait_tensor` calls, wait gaps,
and per-file counts.

Current async/static graph:

- AOT hash:
  `3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`
- actual `_c10d_functional.all_reduce_` call lines: `1,496`
- actual `wait_tensor.default` call lines: `1,496`
- all `1,496` allreduces are followed by a matching wait exactly two lines
  later.
- eight generated Inductor files each contain `187` actual allreduce/wait
  pairs.

Command:

```bash
scripts/classify-vllm-aot-collectives.py \
  /mnt/fast-ai/vllm-cache-exp/minimax-inductor-partition-compile1-p512n512-20260512T122329Z/torch_compile_cache/torch_aot_compile/3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846/inductor_cache \
  --json > data/minimax-m27-aot-collective-classification-20260513.json
```

The source-level implication is unchanged but clearer: runtime flags are mostly
exhausted, and the remaining large win requires changing the graph/kernels
around these collective fences without changing Q/K variance allreduce,
residual math, expert routing, or target verification.
