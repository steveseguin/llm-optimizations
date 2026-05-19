# MiniMax AR+RMS C10D Repeatability Patch Note, 2026-05-19

This patch note records the rejected AR+RMS fused helper follow-up.

## Runtime Fix Tested

The helper calls `c10d::all_reduce(input, "sum", group_name)` from
`torch/csrc/distributed/c10d/Functional.hpp`, so the model call site must pass
the c10d registered process group name:

```diff
- get_tp_group().unique_name,
+ get_tp_group().device_group.group_name,
```

This fixed the previous runtime failure:

```text
RuntimeError: Could not resolve the process group registered under the name tp:0
```

## Rejection

After the group-name fix, one strict screen passed, but a later fresh-label run
failed `raw145-n256-exact` with a combined token hash mismatch before throughput
benchmarking. The candidate remains default-off and should not be used for
published speed numbers.

## Files Involved

- `vllm/model_executor/models/minimax_m2.py`
- `experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp`
- `experiments/minimax_ar_fused_rms_xpu/__init__.py`
- `benchmarks/b70_minimax_ar_fused_rms_op_smoke.py`
