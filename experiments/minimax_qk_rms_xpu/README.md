# MiniMax Q/K RMS XPU helper

Standalone SYCL/PyTorch extension for testing a MiniMax M2.7 Q/K RMSNorm helper
without replacing the release `vllm_xpu_kernels._C` wheel.

The earlier in-tree helper proved numerically correct, but benchmarking it by
swapping the whole `_C.abi3.so` regressed the rest of vLLM's XPU kernels. This
extension keeps the release wheel intact and exposes only:

- `minimax_qk_rms_xpu.var(qkv, qk_var, q_size, kv_size)`
- `minimax_qk_rms_xpu.apply(qkv, qk_var, q_weight, k_weight, q_out, k_out, q_size, kv_size, eps)`

Build:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/setvars.sh --force
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MINIMAX_QK_RMS_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_QK_RMS_XPU_SYCL_DEVICE=bmg
python -m pip install -e /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu
```

Use in vLLM only after applying the companion MiniMax patch and setting:

```bash
export VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
```

Status on 2026-05-09: technically works, but benchmarked slower than the stock
vLLM Q/K RMS path. Keep disabled unless re-testing a modified kernel.
