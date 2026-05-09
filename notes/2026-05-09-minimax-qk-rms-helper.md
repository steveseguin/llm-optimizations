# MiniMax M2.7 Q/K RMS helper experiment

Target: `Lasimeri/MiniMax-M2.7-int4-AutoRound`, vLLM/XPU TP4 on 4x Arc Pro B70, FP16 activations, llm-scaler raw-u4 decode MoE path, `INPUT_LEN=512`, `MAX_MODEL_LEN=2048`, `MAX_BATCHED_TOKENS=1024`, batch 1.

Baseline to beat:

- Best p512/n256 default-IPC u4-decode run: `34.578045 tok/s` output, log `vllm-minimax-m27-autoround-tp4-p512n256-20260509T112853Z.log`.
- Restored release-wheel control after the first helper attempt: `33.444226 tok/s` output, log `vllm-minimax-m27-autoround-tp4-p512n256-20260509T164548Z.log`.

Why this was tested:

- Previous timing showed `minimax.attn.qk_norm` around `0.25 ms` per layer with sync timing enabled, larger than attention and comparable to post-u4 MoE.
- The first in-tree helper was numerically correct, but swapping the whole development `_C.abi3.so` over the v0.1.7 wheel regressed helper-off p512/n256 to `32.794741 tok/s`.
- This follow-up keeps the release `vllm_xpu_kernels._C` wheel intact and builds a separate `minimax_qk_rms_xpu` extension.

Implementation artifacts:

- Standalone extension: `experiments/minimax_qk_rms_xpu/`
- vLLM hook patch: `patches/vllm-minimax-qk-rms-xpu-helper-20260509.patch`
- original in-tree helper patch: `patches/vllm-xpu-kernels-minimax-qk-rms-helper-20260509.patch`

Build notes:

- Build isolation must be disabled so `setup.py` can see the active XPU PyTorch.
- The extension must be built with the 2025.3 compiler headers/libraries to match `libsycl.so.8`.
- Plain pybind calls fail under vLLM/TorchDynamo during memory profiling. Registering real `torch.ops.minimax_qk_rms_xpu.var/apply` custom ops plus fake registrations fixes tracing.
- Generic SPIR64 launches correctly but is slower; BMG AOT with `MINIMAX_QK_RMS_XPU_SYCL_TARGETS=spir64_gen,spir64` and `MINIMAX_QK_RMS_XPU_SYCL_DEVICE=bmg` launches correctly.

Correctness:

- Standalone custom-op test passes on XPU for FP16 and BF16.
- FP16 max differences against PyTorch reference: variance `1.19e-7`, Q `4.88e-4`, K `0.0`.
- BF16 max differences: variance `1.19e-7`, Q `0.0`, K `0.0`.

Benchmark results:

| Variant | Log | Output tok/s | Notes |
| --- | --- | ---: | --- |
| Pybind helper | `20260509T170321Z.log` | n/a | Failed during Dynamo trace; raw pybind is not graph-safe. |
| Generic SPIR64 custom op | `20260509T171226Z.log` | `22.5659` | Correct but much slower than baseline. |
| BMG AOT custom op, all token counts | `20260509T172225Z.log` | `32.5460` | Runs, but below restored control and best baseline. |
| BMG AOT custom op, decode-only `<=4` tokens | `20260509T173058Z.log` | `23.8984` | Worse; decode-only gating does not rescue this helper. |

Decision:

Do not promote this helper. Keep it as a reproducibility artifact and leave `VLLM_MINIMAX_QK_RMS_XPU_HELPER` unset for normal runs. The stock vLLM/Inductor Q/K RMS path is better on this stack. The remaining Q/K cost is more likely allreduce and scheduling latency than local RMS arithmetic; a useful future path would need communicator-level fusion or a true XPU analogue of vLLM's CUDA `minimax_allreduce_rms_qk`, not a standalone two-kernel helper.

Speculative decode note:

For this MiniMax random single-session harness, n-gram speculation remains a poor fit. Prior measured MiniMax results were `2.26 tok/s` output for CPU n-gram and `3.15 tok/s` for GPU n-gram on p64/n16, while the non-speculative path was already far faster. The AutoRound checkpoint also does not provide a usable MiniMax MTP/draft head in the current vLLM path. Keep speculation work focused on Qwen FP8/Q4 or on a real MiniMax draft/MTP checkpoint if one becomes available.

