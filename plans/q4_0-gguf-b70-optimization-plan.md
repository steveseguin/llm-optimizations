# Qwen3.6 27B Q4_0 GGUF B70 Optimization Plan

Date: 2026-05-03

## Position

The INT4 AutoRound vLLM result is interesting but does not count as success against the original quality-preserving target. For apples-to-apples comparison, the working target remains Qwen3.6 27B `Q4_0` GGUF on llama.cpp.

Current local single-B70 GGUF baselines:

- SYCL graph enabled, oneDNN enabled, Q4_0/f16 KV: `24.553-24.570 tok/s` stable warmup-enabled result.
- SYCL graph enabled, oneDNN disabled, Q4_0/f16 KV: `24.249 tok/s` stable warmup-enabled result.
- Vulkan, patched B70 core count, system Mesa: `22.19 tok/s`.

Current host state after the latest crash sequence:

- SYCL/Level Zero should be treated as wedged until reboot or equivalent driver recovery.
- Fresh single-card sanity runs now fail with Level Zero OOM in `MUL_MAT`, including the previously working AOT+DNN binary.
- PCI reset completed but did not recover the stack; `sycl-ls` then stuck in kernel `D` state and dmesg reported Xe TLB invalidation timeout/runtime suspend errors.

Comparable external B70 Q4_0 results:

- Windows Vulkan: up to `28.8 tok/s` on LocalMaxxing.
- Windows SYCL command shape submitted by Steve: `27.03 tok/s`.
- Linux Vulkan with Mesa main ANV, B70 core-count patch, graphics queue allowed, flash attention off: `25.16 tok/s`.

Therefore the Linux target is not speculative: first reach `>=27 tok/s`, then `>=29 tok/s`, without changing the model quantization.

## Current Signals

- Intel compute-runtime `26.14.37833.4` is still the latest GitHub release found today and is already installed. It includes Level Zero `v1.28.2`, IGC `v2.32.7`, and gmmlib `22.9.0`.
- Local llama.cpp is at `b9010`/`d05fe1d`; `origin/master` is one commit ahead at `db44417`.
- Upstream llama.cpp `origin/master` still lacks the Intel Arc Pro B70 Vulkan core-count entry for PCI ID `0xE223`, so the local Vulkan B70 patch remains required.
- In current upstream llama.cpp, Vulkan's Intel Q4_0 MMVQ disable is still correct for B70 single-token decode: forcing MMVQ dropped Q4_0 decode to about `10.35 tok/s`.
- SYCL graph mode is important on this host. `GGML_SYCL_DISABLE_GRAPH=0` improved single-B70 Q4_0 from the first `19.17 tok/s` run to a stable `24.249 tok/s`.
- oneDNN enabled is a small but repeatable single-card gain:
  - non-AOT oneDNN build: `24.553 tok/s`;
  - BMG-G31 AOT plus oneDNN build: `24.570 tok/s`.
- BMG-G31 AOT by itself is valid after the CMake flag fix, but it only adds about `0.09 tok/s` versus the non-AOT DNN-disabled build.
- Existing SYCL debug logs confirm single-card Q4_0 decode already uses reordered MMVQ:
  - one-token debug run had `344` reordered Q4_0 MMVQ calls and no plain Q4_0 MMVQ calls.
  - Therefore the next single-card work should target kernel/runtime efficiency rather than assuming reorder is missing.
- `GGML_SYCL_PRIORITIZE_DMMV=1` is currently unsafe on the combined AOT+DNN build: it produced a segmentation fault and an empty JSONL result.
- SYCL row split had real source issues:
  - split-buffer factory signature did not match the backend ABI;
  - split-buffer metadata did not model CUDA's per-main-device split buffer type;
  - split-buffer init only had one queue pointer but indexed by device;
  - reorder optimization tried to use dummy split-buffer base pointers.
- After fixing those local SYCL row-split issues, row mode starts generation but is still unusably slow: a 128-token smoke run timed out after 300 seconds.
- Dual tensor debug with `-fa 1` preserves the reordered Q4_0 path but introduces heavy copy traffic:
  - one-token debug run logged `113` explicit SYCL copy calls and `48` memcpy-path copies before Level Zero OOM under debug.
  - This supports the working theory that tensor split is copy/synchronization bound, not missing the Q4_0 matvec kernel.
- OpenVINO 2026.1 docs list an internal `GatedDeltaNet` operation, but the local OpenVINO 2026.1.2 source tree does not expose a matching implementation symbol under `src/`. This is worth investigating, but OpenVINO remains an R&D track until the recurrent Qwen3.6 path can stay on GPU.

## Quality Constraints

- Do not count INT4 AutoRound, Q4_K_M, Q8 KV, or draft/speculative approximations as equivalent to the Q4_0 GGUF baseline unless they are explicitly marked as separate quality tradeoff results.
- Default benchmark mode keeps `Q4_0` weights and `f16` KV cache.
- Submit LocalMaxxing results only when the payload clearly states the exact model, quantization, backend, command, and whether any quality-affecting option was used.
- Add quality checks before treating any non-GGUF speed path as a replacement:
  - deterministic greedy generation diff on fixed prompts;
  - perplexity or logprob spot check where llama.cpp supports it;
  - small task eval set for instruction following and coding prompts.

## Immediate Benchmark Harness

Created:

- `/home/steve/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`
- `/home/steve/bench-qwen36-q4_0-gguf-sycl-matrix.sh`

Both write JSONL plus metadata under `/home/steve/bench-results/qwen36-q4_0-gguf/`.

Initial matrix priorities:

- Vulkan:
  - system Mesa versus locally built Mesa main ANV;
  - compute queue versus `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`;
  - flash attention `0/1`;
  - `--poll 0/50/100`;
  - `-ub 64/128/256/512`;
  - keep `-ctk f16 -ctv f16`.
- SYCL:
  - exact Windows comparison command shape, including `-pg`/`-d` style tests;
  - oneDNN on/off via `GGML_SYCL_DISABLE_DNN`;
  - optimization/reorder on/off via `GGML_SYCL_DISABLE_OPT`;
  - flash attention `0/1`;
  - `-ub 64/128/256/512`;
  - keep `-ctk f16 -ctv f16`.

## Work Tracks

### Track A: Vulkan to 27 tok/s

Goal: make Linux Vulkan match the known Windows Q4_0 range.

Steps:

1. Rebuild llama.cpp from `origin/master` plus the `0xE223 -> 32 cores` patch.
2. Run the Vulkan matrix on system Mesa to refresh the baseline.
3. Build Mesa main ANV locally and select it via `VK_DRIVER_FILES` / `VK_ICD_FILENAMES`, matching the public Linux result setup.
4. Re-run the same matrix with Mesa main.
5. If Mesa main reaches ~25 tok/s but not 27+, instrument Vulkan dispatch:
   - confirm B70 core-count and split-k decisions;
   - add an env override for Intel shader core count/split-k to sweep without recompiling;
   - inspect whether Q4_0 matvec or GatedDeltaNet/SSM kernels dominate decode time.
6. If graphics queue is only faster on Mesa main, isolate whether the delta is queue family selection, cooperative matrix behavior, or synchronization.

### Track B: SYCL to 27 tok/s

Goal: reproduce or beat the Windows SYCL `27.03 tok/s` Q4_0 result on Linux.

Steps:

1. Reboot or otherwise recover the Xe/Level Zero runtime before running more SYCL GPU benchmarks.
2. After recovery, start with `sycl-ls`, then a one-token/small-token single-card sanity run before any matrix.
3. Keep `GGML_SYCL_DISABLE_GRAPH=0` as the current best single-card path.
4. Re-run the known Windows command shape where possible against the clean `db44417` build.
5. Keep oneDNN enabled for the current best reproducible single-card run.
6. Keep BMG-G31 AOT as a reproducible build variant, but do not expect it alone to reach the Windows target.
7. Avoid `GGML_SYCL_PRIORITIZE_DMMV=1` until the segfault is isolated.
8. Instrument or profile kernel time for the reordered Q4_0 MMVQ path and recurrent ops:
   - `reorder_mul_mat_vec_q4_0_q8_1_sycl`;
   - `quantize_row_q8_1_sycl`;
   - `GATED_DELTA_NET`;
   - `SSM_CONV`;
   - recurrent state copy/update paths.
9. Compare Linux oneAPI/Level Zero runtime behavior against the known Windows SYCL result; the reorder path is not the missing piece.

### Track C: Dual B70 Without Quality Loss

Goal: only return to dual-GPU GGUF once the single-card path is healthy.

Known blockers:

- Vulkan tensor split is currently slower than single-card.
- Vulkan layer split only matches single-card because decode is serial through layers.
- SYCL layer split is stable but only matches single-card: `23.978 tok/s` versus `24.249 tok/s` single-card.
- SYCL tensor split is stable with flash attention but slower: `19.524 tok/s` on a 128-token smoke run.
- SYCL row split now reaches generation after local fixes, but the split matmul/copy path is far too slow: 128-token smoke timed out after 300 seconds.
- After the latest experimental split safety rebuild, row split still timed out on a 16-token smoke run, and tensor split regressed to Level Zero OOM at the final `output.weight` projection.
- SYCL tensor split debug shows many small copies in the decode step, including recurrent state copies. A one-token debug run saw `113` SYCL copy calls before Level Zero OOM under debug.
- SYCL row split had a direct matmul pointer bug: split weights must use `src0_extra->data_device[i]`, not dummy `src0->data`.
- SYCL recurrent kernels that directly dereference `tensor->data` must not accept split-buffer inputs until they are made split-aware.

Steps:

1. Do not run more dual-B70 SYCL tests until the current driver wedge is cleared.
2. Fix single-card Q4_0 first; current best is `24.570 tok/s`, target remains `>=27`.
3. Treat layer split as a memory-capacity/throughput-for-multiple-sessions path, not a single-session acceleration path.
4. Keep the current SYCL split safety edits marked experimental, not accepted:
   - use per-device split tensor pointers for split matmul;
   - keep `SSM_CONV` and `GATED_DELTA_NET` away from split buffers.
5. Split the dual-GPU work into two separate investigations:
   - row split: fix correctness and then remove serial activation broadcast/gather waits;
   - tensor/meta split: fix the final projection OOM/regression and reduce copy/sync costs.
6. For tensor split, avoid blind split-ratio sweeps until the final `output.weight` OOM is understood.
7. For Vulkan tensor split, profile synchronization and per-op ownership instead of sweeping blind flags.
8. Revisit true dual-B70 speedup through speculative/draft batching only after the non-speculative single-card backend reaches the Windows range.

### Track D: OpenVINO R&D

Goal: determine whether OpenVINO can become a quality-preserving GGUF backend for Qwen3.6.

Steps:

1. Resolve the `GatedDeltaNet` doc/source mismatch:
   - check if the op exists in a newer OpenVINO branch;
   - check if it is plugin-private or generated outside the visible source tree;
   - check whether OpenVINO GenAI packages expose it.
2. If an implementation exists, write a llama.cpp OpenVINO translator for `GGML_OP_GATED_DELTA_NET` targeting it.
3. If not, keep OpenVINO work focused on reducing graph split count but do not expect near-term 27 tok/s.

## Success Criteria

- First accepted improvement: Q4_0 GGUF single B70 `>=27 tok/s` with a reproducible command and no quality-changing flags.
- Strong success: Q4_0 GGUF single B70 `>=29 tok/s`.
- Dual-card success: Q4_0 GGUF single session `>=48 tok/s` first, then `>=52 tok/s`, without switching away from Q4_0.
