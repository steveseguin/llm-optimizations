# MiniMax M2.7 Exact-Shape XCCL Allreduce Microbench

Date: 2026-05-19

## Why

After allreduce shape logging showed MiniMax decode collectives at exact shapes `(1, 2)` FP32 for Q/K variance and `(1, 3072)` FP16 for hidden/residual paths, I added a focused XCCL microbench to measure the raw collective cost on the four B70s.

This separates two questions:

- Is the raw XCCL collective itself too slow?
- Or is the model losing time at framework, graph, clone/copy, and scheduling boundaries around otherwise small collectives?

## Command

```bash
CCL_TOPO_P2P_ACCESS=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZE_AFFINITY_MASK=0,1,2,3 \
B70_MINIMAX_AR_ITERS_SCALE=4 \
torchrun --standalone --nproc-per-node=4 \
  /home/steve/llm-optimizations-publish/benchmarks/b70_xccl_minimax_allreduce_shapes.py
```

## Key Results

Long confirmation run: `/home/steve/bench-results/collectives/b70-minimax-allreduce-shapes-long-20260519T042421Z.clean.json`

Mean latency by shape:

- Q/K decode FP32 `(1, 2)`: in-place `0.016774 ms`, clone `0.020703 ms`, empty-copy `0.021564 ms`
- Q/K prompt/profile FP32 `(512, 2)`: in-place `0.014893 ms`, clone `0.019472 ms`
- hidden decode FP16 `(1, 3072)`: in-place `0.014878 ms`, clone `0.019358 ms`
- hidden prompt/profile FP16 `(512, 3072)`: in-place `0.117967 ms`, clone `0.120412 ms`

For decode-sized collectives, raw XCCL allreduce is roughly `15-17 us` in-place. Clone/copy setup adds roughly `4-5 us` for the small tensors.

## Interpretation

The raw collectives are not free, but the microbench numbers are lower than the full-model penalty implied by broad allreduce-threshold experiments. That makes the next optimization target more specific:

- avoid pushing hidden-state collectives into worse compiler paths;
- keep alias-correct tiny FP32 in-place allreduce for `(1, 2)`;
- avoid generic `numel <= 2048` or `numel <= 4096` thresholds;
- focus on boundary fusion around Q/K variance allreduce, residual allreduce, and MoE epilogues rather than CCL flag-only reties.

This also explains why `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=4096` regressed: it captured the `(1, 3072)` FP16 hidden decode allreduce. The `n2048` screen avoided that hidden tensor but still lost versus the tiny-FP32-only path, so dtype/shape-specific fusion remains the better route.

## Artifacts

- Script: `benchmarks/b70_xccl_minimax_allreduce_shapes.py`
- First run raw JSON: `/home/steve/bench-results/collectives/b70-minimax-allreduce-shapes-20260519T042325Z.json`
- First run clean JSON: `/home/steve/bench-results/collectives/b70-minimax-allreduce-shapes-20260519T042325Z.clean.json`
- Long run raw JSON: `/home/steve/bench-results/collectives/b70-minimax-allreduce-shapes-long-20260519T042421Z.json`
- Long run clean JSON: `/home/steve/bench-results/collectives/b70-minimax-allreduce-shapes-long-20260519T042421Z.clean.json`
- Local data: `data/minimax-m27-xccl-exact-shape-microbench-20260519.json`
