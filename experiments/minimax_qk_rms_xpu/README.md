# MiniMax Q/K RMS XPU helper

Standalone SYCL/PyTorch extension for testing a MiniMax M2.7 Q/K RMSNorm helper
without replacing the release `vllm_xpu_kernels._C` wheel.

The earlier in-tree helper proved numerically correct, but benchmarking it by
swapping the whole `_C.abi3.so` regressed the rest of vLLM's XPU kernels. This
extension keeps the release wheel intact and exposes only:

- `minimax_qk_rms_xpu.var(qkv, qk_var, q_size, kv_size)`
- `minimax_qk_rms_xpu.apply(qkv, qk_var, q_weight, k_weight, q_out, k_out, q_size, kv_size, eps)`
- `minimax_qk_rms_xpu.apply_qk_rope(q, k, qk_var, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out, head_size, rotary_dim, eps, is_neox)`

Build:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MAX_JOBS=1
export MINIMAX_QK_RMS_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_QK_RMS_XPU_SYCL_DEVICE=bmg
python -m pip install --no-build-isolation -e /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu
```

Do not build with `/opt/intel/oneapi/setvars.sh` while `compiler/latest` points
at oneAPI 2026.0 for the current PyTorch XPU runtime. That produced an import
failure on 2026-05-10 with an undefined `sycl::queue::submit_with_event_impl`
symbol. Building with the compiler-specific 2025.3 env and running without
manually sourcing oneAPI matched the venv-provided `libsycl.so.8`.

Use in vLLM only after applying the companion MiniMax patch and setting:

```bash
export VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
```

Status on 2026-05-09: technically works, but benchmarked slower than the stock
vLLM Q/K RMS path. Keep disabled unless re-testing a modified kernel.

Status on 2026-05-10: `apply_qk_rope` is numerically valid as a small standalone
FP16 helper, but the vLLM compiled p512/n512 path regressed from `39.610585` to
`35.681825` output tok/s when enabled with
`VLLM_MINIMAX_QK_APPLY_ROPE_XPU_HELPER=1`. It remains default-off.

Plain `var`/`apply` was retested after the AOT-cache regression work. A fresh
isolated p512/n512 compile reproduced the cold KV-cache artifact (`9,408` GPU KV
tokens, `28.664` output tok/s), while warmed reloads reached `35.722` output
tok/s at p512/n512 and `36.572` output tok/s at p512/n1536. This is still below
the stock path and the accepted `41.130667` MiniMax high, so
`VLLM_MINIMAX_QK_RMS_XPU_HELPER` should also stay unset.
