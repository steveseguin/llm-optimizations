# MiniMax Q/K Apply+RoPE Helper Negative, 2026-05-10

## Goal

Test a lower-risk XPU helper before attempting a full Lamport-style XPU Q/K allreduce. The helper preserves the existing vLLM Q/K variance computation and TP allreduce, then fuses only the post-allreduce RMS application with RoPE for tiny decode batches.

This is intentionally default-off:

- `VLLM_MINIMAX_QK_APPLY_ROPE_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_APPLY_ROPE_XPU_HELPER_MAX_TOKENS=4` by default.

## Implementation

Code:

- Standalone extension: `experiments/minimax_qk_rms_xpu/minimax_qk_rms_xpu.cpp`
- vLLM hook patch: `patches/vllm-minimax-qk-apply-rope-xpu-helper-20260510.patch`

New op:

- `torch.ops.minimax_qk_rms_xpu.apply_qk_rope(q, k, qk_var, q_weight, k_weight, positions, cos_sin_cache, q_out, k_out, head_size, rotary_dim, eps, is_neox)`

The op consumes FP32 `q` and `k`, allreduced FP32 `[num_tokens, 2]` variance, FP16/BF16 RMS weights, positions, and the RoPE cache. It writes FP16/BF16 `q_out` and `k_out`.

## Build Note

The first rebuild used `/opt/intel/oneapi/setvars.sh --force`, where `compiler/latest` points to oneAPI 2026.0. That produced an import failure:

```text
undefined symbol: sycl::queue::submit_with_event_impl...
```

Rebuilding with the compiler-specific 2025.3 environment fixed the extension:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MAX_JOBS=1
export MINIMAX_QK_RMS_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_QK_RMS_XPU_SYCL_DEVICE=bmg
python -m pip install --no-build-isolation -e /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu
```

Runtime import should be tested from the normal venv environment without manually sourcing oneAPI:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
python -c "import torch, minimax_qk_rms_xpu; print(torch.xpu.device_count())"
```

## Validation

Standalone FP16 numeric check:

- `q_max_diff = 0.001953125`
- `k_max_diff = 0.001953125`

BF16 smoke also executed and returned finite outputs.

vLLM smoke:

- Shape: p32/n8
- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p32n8-20260510T043923Z.log`
- Completed without worker crash. Not a performance result because compile/load dominates the tiny shape.

## Performance

Baseline p512/n512 fast-NVMe FP16 path:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T011816Z.log`
- Total tok/s: `79.22117`
- Output tok/s: `39.610585`

Apply+RoPE helper p512/n512:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T044214Z.log`
- Total tok/s: `71.363649`
- Output tok/s: `35.681825`
- Result: negative, about 9.9% below the baseline.

Eager synchronized timing with the helper is misleadingly encouraging:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n4-20260510T044449Z.log`
- Rank-0 last steady samples showed `minimax.attn.qk_norm` near `0.21 ms/layer` and no separate `minimax.attn.rope` label, versus the earlier default eager sample near `0.465 ms/layer` plus a separate RoPE label.
- The compiled throughput result is the deciding signal. The helper likely blocks a better compiled schedule or changes kernel launch ordering enough to lose more than it saves.

Conclusion:

- Keep `VLLM_MINIMAX_QK_APPLY_ROPE_XPU_HELPER` unset for real benchmarks.
- Do not submit to LocalMaxxing.
- The next Q/K path should be a real XPU peer-memory allreduce/RMS fusion or an inductor-level fusion that does not degrade the compiled graph.

## CCL Direct Screen

Following Intel oneCCL documentation, `CCL_ALLREDUCE=direct` was screened because current MiniMax decode collectives are tiny. Intel documents that non-`topo` GPU-buffer allreduce algorithms copy GPU data to host and then use the CPU algorithm, so this was expected to be risky.

- Source: <https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-15/environment-variables.html>
- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T044829Z.log`
- Total tok/s: `32.290690`
- Output tok/s: `16.145345`
- Result: very negative. Keep `CCL_ALLREDUCE` unset/default `topo`.
