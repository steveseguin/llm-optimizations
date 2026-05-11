# MiniMax M2.7 AutoRound AR+RMS XPU Custom Op Screen, 2026-05-11

## Result

Negative. The C++ custom op path compiled and executed in vLLM, but it cut p512/n512 throughput roughly in half versus the current clean MiniMax AutoRound TP4 reference.

Benchmark:

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: local vLLM `v0.20.1`, XPU, TP4, llm-scaler INT4 MoE path enabled
- Hardware: 4x Intel Arc Pro B70 32GB
- Prompt/generated tokens: p512/n512
- Context length: 1024
- Result: `19.89` output tok/s, `39.78` total tok/s
- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-fused-rms-p512n512/vllm-minimax-m27-autoround-tp4-p512n512-20260511T003052Z.log`
- JSON: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-fused-rms-p512n512/vllm-minimax-m27-autoround-tp4-p512n512-20260511T003052Z.json`

Current comparison points:

- Clean p512/n512 reference: about `35.65-39.61` output tok/s, depending on cache/run condition.
- Accepted long p512/n1536 reference: `37.552538` output tok/s and `50.070051` total tok/s.

This result is not LocalMaxxing-worthy as an achievement. Keep it as a documented negative and reproducibility artifact.

## What Changed

Patch artifact:

- `patches/vllm-minimax-ar-fused-rms-xpu-negative-20260511.patch`

Experiment source:

- `experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp`
- `experiments/minimax_ar_fused_rms_xpu/__init__.py`
- `experiments/minimax_ar_fused_rms_xpu/build.py`
- `benchmarks/b70_minimax_ar_fused_rms_op_smoke.py`

vLLM MiniMax path added an env-gated flag:

- `VLLM_MINIMAX_AR_FUSED_ADD_RMS_XPU=1`

When enabled, `o_proj` skips its normal row-parallel reduce, and the post-attention residual/RMSNorm boundary calls:

```python
torch.ops.minimax_ar_fused_rms_xpu.ar_fused_add_rms(
    hidden_states,
    residual,
    self.post_attention_layernorm.weight.data,
    get_tp_group().device_group.group_name,
    self.post_attention_layernorm.variance_epsilon,
)
```

The correct functional c10d process-group name was important. Passing vLLM's logical `get_tp_group().unique_name` (`tp:0`) failed in compiled vLLM because PyTorch functional collectives expected the registered group name. Passing `get_tp_group().device_group.group_name` generated group `'3'` and fixed the compile/runtime failure.

## Validation

Standalone four-rank XPU smoke passed:

```text
{'out_mean': 1.0, 'residual_mean': 11.0}
```

vLLM p1/n4 smoke after the group-name fix passed:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-fused-rms-smoke3/vllm-minimax-m27-autoround-tp4-p1n4-20260511T002707Z.log`
- Result: `0.631312` total tok/s. This is only a liveness smoke and is dominated by tiny request overhead.

Compiled graph inspection for the p512/n512 run:

- `62` calls to `minimax_ar_fused_rms_xpu.ar_fused_add_rms`
- `125` remaining visible c10d wait sites

The patch therefore changed the graph shape as intended, but the implementation is still an opaque dispatcher path that internally performs c10d allreduce, wait, dtype conversions, add, pow/mean/rsqrt, and multiply. It removed visible graph nodes without creating a true SYCL/Inductor fused kernel.

## Lesson

The useful target is still valid: post-attention allreduce -> residual add -> RMSNorm is a real boundary. The implementation shape is wrong.

Do not spend more time on opaque Python/C++ wrappers around functional allreduce unless the wrapper owns a real device kernel or compiler lowering. The next useful version needs one of:

- a SYCL kernel that consumes the reduced buffer and writes residual/RMS output with fewer launches and no extra dtype churn,
- an Inductor lowering/pass that keeps the collective and following epilogue schedulable,
- or a communication primitive that supports an epilogue directly after the allreduce.

This path should stay default-off. Baseline runs must leave `VLLM_MINIMAX_AR_FUSED_ADD_RMS_XPU` unset.

