# MiniMax Pair Argmax XPU Helper

Default-off experiment for the MiniMax M2.7 TP4 local-argmax tail.

The current quality-promoted vLLM path computes local max logits on each TP rank,
all-gathers `(float32 value, float32 global_token_id)` pairs, then uses PyTorch
ops to reduce the gathered pairs. This helper preserves the same math:

1. fill a local FP32 pair tensor on XPU from local max values and int64 token ids;
2. use `c10d::all_gather_into_tensor`;
3. reduce the gathered pairs to int64 token ids with one tiny SYCL kernel.

It intentionally does not use `ReduceOp.MAX`, because the B70/XCCL packed-argmax
probe showed reduction-op correctness problems for this use case.

Build:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MAX_JOBS=1
export MINIMAX_PAIR_ARGMAX_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_PAIR_ARGMAX_XPU_SYCL_DEVICE=bmg
python -m pip install --no-build-isolation -e /home/steve/llm-optimizations-publish/experiments/minimax_pair_argmax_xpu
```

Do not promote until standalone correctness, raw145 exact hashes, semantic
repeatability, and adjacent p512/n1536 control comparison all pass.
