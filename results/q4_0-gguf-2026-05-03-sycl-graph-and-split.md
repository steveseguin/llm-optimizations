# Qwen3.6 27B Q4_0 GGUF B70 Results - 2026-05-03

Host: Ubuntu 24.04.4 LTS, AMD EPYC 9015 8-Core Processor, 16 logical CPUs, 16 GiB RAM, 2x Intel Arc Pro B70 / BMG G31 32 GB.

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

llama.cpp worktree: `/home/steve/src/llama.cpp-q4-b70`, upstream `db44417` plus local B70 Vulkan and SYCL split-buffer patches.

## Single B70

Best quality-preserving Linux result so far:

- Backend: llama.cpp SYCL / Level Zero.
- Selector: `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- Flags: `GGML_SYCL_DISABLE_GRAPH=0`, `GGML_SYCL_DISABLE_OPT=0`, `GGML_SYCL_DISABLE_DNN=1`.
- Command shape: `llama-bench -dev SYCL0 -ngl 99 -p 0 -n 512 -sm none -b 512 -ub 64 -ctk f16 -ctv f16 -t 8 -fa 0 -r 3` with warmup enabled.
- Result: `24.249 tok/s`, samples `24.2552`, `24.2452`, `24.2468`.
- Quality status: no model quantization change; Q4_0 weights and f16 KV.

Reference Vulkan result on system Mesa 25.2.8:

- Backend: llama.cpp Vulkan.
- Command shape: `llama-bench -dev Vulkan0 -ngl 99 -p 0 -n 512 -sm layer -b 512 -ub 64 -ctk f16 -ctv f16 -t 8 -fa 0 --poll 50`.
- Result: `22.19 tok/s`.

## Vulkan Experiments

- `GGML_VK_FORCE_MMVQ=1` dropped Q4_0 decode to about `10.35 tok/s`; upstream's Intel Q4_0 MMVQ disable remains correct for this B70 decode shape.
- `GGML_VK_INTEL_XE2_DMMV_LARGE_MAX_M` threshold sweep over `8192`, `16384`, `32768`, `65535` peaked around `22.16 tok/s`, within noise of baseline.
- Flash attention was slightly slower at this short-context generation-only shape.

## Dual B70

Stable layer-split baselines:

- Vulkan layer split: `-dev Vulkan0/Vulkan1 -sm layer -ts 1/1`, `21.966 tok/s`, samples `21.9417`, `21.9671`, `21.9904`.
- SYCL layer split: `ONEAPI_DEVICE_SELECTOR=level_zero:* -dev SYCL0/SYCL1 -sm layer -ts 1/1`, `23.978 tok/s`, samples `23.675`, `24.11`, `24.1482`.

Conclusion: layer split is stable but does not improve single-session decode because token generation is still serial through the layer stack.

Tensor split:

- SYCL tensor split requires flash attention.
- Smoke result: `-n 128 -sm tensor -ts 1/1 -fa 1`, `19.524 tok/s`.
- Below single-card; not a current route to the 80% dual-card speedup target.

Row split:

Local correctness fixes applied in `patches/llama-cpp-db44417-b70-sycl-split-fixes.patch`:

- Fix SYCL split-buffer factory ABI to `(int main_device, const float * tensor_split)`.
- Add per-main-device split buffer metadata/name/device registration.
- Make SYCL device `supports_buft` accept SYCL split buffers for their owning backend device.
- Initialize one split-buffer queue pointer per SYCL device.
- Guard Q4_0 reorder optimization away from SYCL split buffers because split tensors use dummy base pointers and real per-device pointers in tensor extras.

After those fixes, SYCL row split reaches generation, but a 128-token smoke run timed out after 300 seconds without producing a result. Row split is therefore correctness-improved but still unusable for the single-session performance goal.

## Next Actions

- Keep SYCL graph-enabled single-card as the current best Q4_0 Linux path.
- Chase the remaining single-card gap to Windows `27+ tok/s` before expecting dual-card scaling.
- Build/test Mesa main ANV to reproduce the public Linux Vulkan `25.16 tok/s` result.
- Profile SYCL row split only if we decide to invest in split matmul/copy redesign; current behavior is too slow for blind flag sweeps.
