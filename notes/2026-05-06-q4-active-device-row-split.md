# 2026-05-06 Q4 active-device row split follow-up

## Change

Patched `src/llama-model.cpp` in `/home/steve/src/llama.cpp-q4-b70` so `LLAMA_SPLIT_MODE_ROW` split-buffer creation maps llama.cpp's selected device list back to physical backend-registry indices before calling `ggml_backend_split_buffer_type`.

The previous path passed `params.tensor_split` directly to the backend. That array is indexed by selected model-device order, while SYCL split buffers interpret it as global physical SYCL device order. With four visible B70s, a target model using only a subset could accidentally give nonzero row ranges to an unselected card.

Focused patch artifact:

- `/home/steve/llm-optimizations/patches/llama-cpp-active-device-row-split-current-20260506.patch`
- note: the artifact is the current `src/llama-model.cpp` diff from the local tree, so it also includes the already-active Qwen recurrent split-state context in that file; the new row-split accounting hunk starts at `make_backend_reg_tensor_split`.

## Validation

- Build passed:
  - `cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench llama-cli -j 8`
- Known-good tensor-split path still works after the change:
  - model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
  - devices: `SYCL2/SYCL1/SYCL3`
  - split mode: `tensor`
  - prompt/decode: `128` prompt, `128` output
  - result: `45.065268 tok/s` decode, `106.106708 tok/s` prefill
  - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sanity-tensor3-after-active-split-20260506T183355Z.log`
  - jsonl: `/home/steve/bench-results/qwen36-q4_0-gguf/sanity-tensor3-after-active-split-20260506T183355Z.jsonl`

## Negative result

A small Qwen3.5 4B row-split smoke on `SYCL2/SYCL3` still aborted inside the existing SYCL split matmul path:

- model: `/home/steve/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`
- command shape: `-dev SYCL2/SYCL3 -ngl 999 -sm row -ts 1/1 -mg 0 -p 32 -n 8`
- failure: Level Zero `UR_RESULT_ERROR_DEVICE_LOST` inside `oneapi::mkl::blas::column_major::gemm` from `ggml_sycl_op_mul_mat_sycl`
- log: `/home/steve/bench-results/active-split/qwen35-4b-row-sycl23-smoke-20260506T183238Z.log`
- jsonl/backtrace: `/home/steve/bench-results/active-split/qwen35-4b-row-sycl23-smoke-20260506T183238Z.jsonl`
- kernel log: `xe 0000:a3:00.0` reset at `2026-05-06 14:32:41 America/Toronto`

Interpretation: active-device split accounting is a real bug and is now patched locally, but row split remains unsafe for throughput experiments because the SYCL split matmul path can still trigger a GPU reset. The production Q4 path remains tensor split, not row split.

Do not submit this diagnostic to LocalMaxxing.
