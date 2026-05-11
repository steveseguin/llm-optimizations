# MiniMax AR+RMS SYCL And Graph Screen, 2026-05-11

## Purpose

Test two quality-preserving MiniMax M2.7 AutoRound speed ideas on the 4x B70
TP4 path:

- replace the post-attention allreduce + residual/RMS boundary with a default-off
  XPU custom op that keeps XCCL allreduce semantics but runs the residual add and
  RMSNorm math in one SYCL kernel;
- request XPU graph capture on the clean baseline to see if vLLM 0.20.1-local can
  now keep the TP4 communication graphable.

The target remains the quality-conservative p512/n1536 anchor:
`37.552538` output tok/s and `50.070051` total tok/s. The short-run p64/n128
reference is about `50.42` total tok/s / `33.61` output tok/s.

## AR+RMS SYCL Prototype

Artifacts:

- extension source:
  `experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp`
- build helper:
  `experiments/minimax_ar_fused_rms_xpu/setup.py`
- standalone smoke:
  `benchmarks/b70_minimax_ar_fused_rms_op_smoke.py`

The standalone four-rank XPU smoke passes with `max_abs_diff=0.0` against a
reference path that does:

1. XCCL allreduce on the input tensor;
2. residual add in FP32;
3. vLLM RMSNorm semantics including the cast before weight multiply.

Build command:

```bash
cd /home/steve/llm-optimizations-publish/experiments/minimax_ar_fused_rms_xpu
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
CC=/opt/intel/oneapi/compiler/2025.3/bin/icx \
/home/steve/.venvs/vllm-xpu/bin/python setup.py build_ext --inplace
```

Runtime note: do not source the oneAPI compiler environment for PyTorch runtime
smokes. That compiler environment hid XPU devices from `torch.xpu`; the normal
vLLM runtime environment sees all four B70s.

## Results

| Screen | Shape | Total tok/s | Output tok/s | Decision |
| --- | --- | ---: | ---: | --- |
| AR+RMS SYCL, first run | p64/n128 | `23.421757` | `15.614504` | cold compile/runtime artifact |
| AR+RMS SYCL, warm | p64/n128 | `47.654099` | `31.769399` | negative |
| XPU graph requested, first run | p64/n128 | `24.291036` | `16.194024` | graph disabled and cold |
| XPU graph requested, warm | p64/n128 | `49.015733` | `32.677156` | neutral/slightly negative |

Logs:

- AR cold:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-fused-rms-sycl-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T022233Z.log`
- AR warm:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-fused-rms-sycl-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T023148Z.log`
- graph requested cold:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-graph1-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T022614Z.log`
- graph requested warm:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-graph1-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T022923Z.log`

The AR AOT artifact visibly contains the custom op; `strings` counted `125`
`minimax_ar_fused_rms_xpu.ar_fused_add_rms` occurrences. So this was not a
false-negative where the env gate failed to activate.

## Interpretation

The result is useful but negative. Fusing only the post-wait arithmetic is not
enough; wrapping the collective plus RMS work as an opaque op is slightly slower
than vLLM/Inductor's native compiled schedule once warm. The next allreduce/RMS
attempt needs to reduce the communication boundary itself, or be lowered inside
the compiler schedule rather than hiding the work behind an opaque extension op.

The XPU graph flag is also not a current path to a TP4 MiniMax win. vLLM logs:

```text
XPU Graph doesn't support capture communication ops, disabling cudagraph_mode.
```

That makes the warm graph-requested result just a normal baseline rerun. Do not
submit these screens to LocalMaxxing; they are below the existing short and long
anchors.

Structured data:

`data/minimax-m27-ar-rms-sycl-and-graph-screen-20260511.json`
