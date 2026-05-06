# Qwen3.6 27B Q4_0 Skip-Allreduce Diagnostic

Date: 2026-05-06

## Goal

Measure how much of Qwen3.6 27B Q4_0 GGUF tensor-parallel decode time is caused by the small f32 allreduce helpers on B70, using an explicit correctness-breaking diagnostic gate inspired by llm-scaler-style `SKIP_ALL_REDUCE` measurements.

This is not a valid model-quality mode and must not be submitted to LocalMaxxing as a leaderboard result.

## Patch

Added `GGML_SYCL_COMM_SKIP_ALLREDUCE=1` in `ggml/src/ggml-sycl/ggml-sycl.cpp`.

Patch artifacts:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-skip-allreduce-current-20260506.patch`;
- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-skip-allreduce-current-20260506.patch.gz.b64`.

Covered backend hooks:

- `ggml_backend_sycl_comm_allreduce_tensor`: keep each local partial result in place and skip cross-device reduction;
- `ggml_backend_sycl_comm_allreduce_add_tensor`: write `local_partial + local_residual` per device;
- `ggml_backend_sycl_comm_allreduce_to_tensor`: copy local partial to local output;
- `ggml_backend_sycl_comm_allreduce_get_rows_tensor`: gather rows from local partial only.

The gate disables `GGML_SYCL_COMM_SKIP_ROOT_READY` interaction so each local path has an explicit stream barrier dependency.

## Results

Same command shape as the quality-preserving Q4_0 fast path, except `GGML_SYCL_COMM_SKIP_ALLREDUCE`.

3x B70 selector `level_zero:2,1,3`, `p512/n128/r1`:

- skip-allreduce: prompt `238.387693 tok/s`, decode `43.698372 tok/s`;
- same-build control: prompt `135.534672 tok/s`, decode `43.991643 tok/s`;
- JSONL skip: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-skipallreduce-triple213-p512n128-r1-20260506T012317Z.jsonl`;
- JSONL control: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-skipcontrol-triple213-p512n128-r1-20260506T012704Z.jsonl`.

4x B70 selector `level_zero:0,1,2,3`, `p512/n128/r1`:

- skip-allreduce: prompt `315.892740 tok/s`, decode `33.341518 tok/s`;
- same-build control: prompt `101.657601 tok/s`, decode `34.124205 tok/s`;
- JSONL skip: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-skipallreduce-quad0123-p512n128-r1-20260506T012444Z.jsonl`;
- JSONL control: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-skipcontrol-quad0123-p512n128-r1-20260506T012813Z.jsonl`.

## Interpretation

Skipping allreduce gives a large prefill speedup, especially at 4x, but does not improve single-token decode. Decode is flat to slightly worse versus same-build controls.

That means the current 3x/4x Q4_0 decode bottleneck is not primarily the tiny f32 allreduce helper cost by itself. The higher-value next work is local Q4/Q8 vector matvec efficiency, dispatch overhead, or amortizing decode with speculative/MTP-style methods. An fp16 allreduce-buffer experiment is now lower priority unless we add a numerical-quality harness, because this diagnostic does not show decode headroom from removing the f32 collective.

## LocalMaxxing

Not submitted. The gate intentionally breaks tensor-parallel correctness and sacrifices output quality.
