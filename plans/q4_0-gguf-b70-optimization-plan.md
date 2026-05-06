# Qwen3.6 27B Q4_0 GGUF B70 Optimization Plan

Date: 2026-05-03

## Position

The INT4 AutoRound vLLM result is interesting but does not count as success against the original quality-preserving target. For apples-to-apples comparison, the working target remains Qwen3.6 27B `Q4_0` GGUF on llama.cpp.

Current local single-B70 GGUF baselines:

- SYCL graph enabled, oneDNN enabled, Q4_0/f16 KV, four-B70 host with one device exposed: `24.723 tok/s`.
- SYCL graph enabled, oneDNN enabled, Q4_0/f16 KV: `24.553-24.570 tok/s` stable warmup-enabled result.
- SYCL graph enabled, oneDNN disabled, Q4_0/f16 KV: `24.249 tok/s` stable warmup-enabled result.
- Vulkan, patched B70 core count, system Mesa: `22.19 tok/s`.

Current four-B70 state after reboot:

- Four B70s enumerate through Level Zero as selectors `0-3`.
- `steve` is in the `render` group.
- Single-card smoke should expose only one GPU, for example `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- Correct llama.cpp multi-GPU device syntax is slash-separated, for example `-dev SYCL0/SYCL1`; comma-separated device lists are separate benchmark cases, not one multi-GPU run.
- Best quality-preserving dual tensor result so far: `26.872 tok/s` with selector `0,3`, `-dev SYCL0/SYCL1`, `-sm tensor -ts 1/1 -fa 1`.
- Current quad tensor result is poor: `16.548 tok/s` on a short 32-token smoke, so 4-way tensor split needs scheduler/copy work before it is useful.

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
- After adding four B70s, the clean AOT+DNN single-card run with only selector `0` visible reached `24.723 tok/s`.
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
- Corrected dual tensor syntax gives a modest real speedup:
  - selector `0,3`, `-dev SYCL0/SYCL1`, `-sm tensor -ts 1/1 -fa 1`: `26.872 tok/s`.
  - This is only about `+8.7%` over single-card, so the dual/quad target still requires deeper scheduling/copy work.
- Asymmetric tensor split ratios currently expose a meta-scheduler bug:
  - `2/1`, `1/2`, `3/2`, and `2/3` abort at `ggml-backend-meta.cpp:1014`.
  - `3/1` and `1/3` run but are slower than `1/1`.
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

1. Keep single-card tests isolated with `ONEAPI_DEVICE_SELECTOR=level_zero:N`.
2. Keep `GGML_SYCL_DISABLE_GRAPH=0` as the current best single-card path.
3. Re-run the known Windows command shape where possible against the clean `db44417` build.
4. Keep oneDNN enabled for the current best reproducible single-card run.
5. Keep BMG-G31 AOT as a reproducible build variant, but do not expect it alone to reach the Windows target.
6. Avoid `GGML_SYCL_PRIORITIZE_DMMV=1` until the segfault is isolated.
7. Instrument or profile kernel time for the reordered Q4_0 MMVQ path and recurrent ops:
   - `reorder_mul_mat_vec_q4_0_q8_1_sycl`;
   - `quantize_row_q8_1_sycl`;
   - `GATED_DELTA_NET`;
   - `SSM_CONV`;
   - recurrent state copy/update paths.
8. Compare Linux oneAPI/Level Zero runtime behavior against the known Windows SYCL result; the reorder path is not the missing piece.

Latest single-card follow-up:

- Selector-2 oneDNN/flash/ubatch sweep at 256 generated tokens stayed around `24.17-24.45 tok/s`.
- Four-build comparison at 256 generated tokens:
  - `aot-dnn`: `24.805 tok/s`;
  - `dnn`: `24.756 tok/s`;
  - `aot`: `24.561 tok/s`;
  - current base build: `24.498 tok/s`.
- Conclusion: the Windows `>27 tok/s` gap is not explained by oneDNN, flash attention, ubatch, or current AOT variants. Next work should profile reordered Q4_0 matvec dispatch and driver/runtime scheduling.

### Track C: Dual B70 Without Quality Loss

Goal: get tensor-parallel GGUF decode scaling without changing model quality.

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
- Quad tensor split is currently much slower than dual/single.
- Asymmetric tensor split ratios hit a meta-scheduler assertion at `ggml-backend-meta.cpp:1014`.
- The first software copy-path patch changed the best dual Q4_0 result from `26.872 tok/s` to `37.690 tok/s` with no model-quality change.
- Quad tensor with the same patch improved from `16.548 tok/s` to `22.535 tok/s`, but is still slower than single card and dual card.
- Pair sweep with async copy shows all two-GPU selector pairs are close, roughly `37.08-37.35 tok/s` at 128 generated tokens.
- Before the split-anchor fix, all three-GPU tensor splits aborted at `ggml-backend-meta.cpp:1014`, even with equal `-ts 1/1/1`.
- Fixed the equal 3-way Qwen recurrent split planner by anchoring recurrent QKV/gate projections to `ssm_out.weight`; best 3-way 512-token run is now `38.365 tok/s`.
- 4-way tensor improved to `26.366 tok/s` at 128 tokens after the split-anchor fix, but remains non-viable for speed.
- Added experimental SYCL Meta comm hooks with `GGML_SYCL_COMM_ALLREDUCE=1`; best 2-way 512-token run is now `38.621 tok/s` with `-ub 32`.
- Added experimental 2-4 way single-kernel SYCL allreduce with `GGML_SYCL_COMM_SINGLE_KERNEL=1`; best 2-way 512-token run is now `39.849 tok/s` and best 3-way 512-token run is now `41.737 tok/s`, both with `-ub 32`.
- Low-level four-device SYCL peer-read/peer-write stress test passes for the single-kernel pattern, but the path remains env-gated because it is more aggressive than device-to-device copy plus local sum.
- Direct all-reduce does not yet solve 4-way scaling:
  - all-gather-style direct reduction: 3 GPUs `37.069 tok/s`, 4 GPUs `27.387 tok/s`;
  - butterfly-style direct reduction: 3 GPUs `37.547 tok/s`, 4 GPUs `26.836 tok/s`.
- Generalized single-kernel allreduce improves 3/4-way but 4-way remains non-viable for speed:
  - 3 GPUs: `41.737 tok/s` over 512 tokens, 3 repeats with selector order `2,1,3`;
  - 4 GPUs: `31.482 tok/s` over 512 tokens, 1 repeat.
- Follow-up sweeps:
  - 3-GPU `-ub 128` won a short sweep but lost the full 512-token validation; keep `-ub 32`.
  - 3-GPU `--poll 50` remains best.
  - 4-GPU root ordering only moves `30.76-31.31 tok/s`; the 4-way issue is algorithmic fanout/synchronization, not root selection.
- Layer split is confirmed not useful for single-session speed with 2/3/4 B70s: `24.291`, `23.803`, and `23.315 tok/s`.
- Asymmetric 2-way tensor splits are stable after the Qwen anchor fix but much slower than `1/1`.
- GPU power limit changes are out of scope for now; the current path should remain software-only unless the user explicitly changes that constraint.

Steps:

1. Treat layer split as a memory-capacity/throughput-for-multiple-sessions path, not a single-session acceleration path.
2. Use tensor split, not layer split, for the current multi-card speed path.
3. Current best tensor command shape: `ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 ... GGML_SYCL_ASYNC_CPY_TENSOR=1 GGML_SYCL_COMM_ALLREDUCE=1 GGML_SYCL_COMM_SINGLE_KERNEL=1 -dev SYCL0/SYCL1/SYCL2 -sm tensor -ts 1/1/1 -fa 1 -ub 32`.
4. Keep the current SYCL split safety edits marked experimental, not accepted:
   - use per-device split tensor pointers for split matmul;
   - keep `SSM_CONV` and `GATED_DELTA_NET` away from split buffers.
5. Split the dual/quad-GPU work into separate investigations:
   - row split: fix correctness and then remove serial activation broadcast/gather waits;
   - tensor/meta split: reduce copy/sync costs and fix asymmetric split assertions;
   - 4-way tensor split: identify why overhead dominates and why it regresses to `16.5 tok/s`.
6. For tensor split, keep `-ts 1/1` until the meta assertion is fixed.
7. Keep `GGML_SYCL_ASYNC_CPY_TENSOR=1` as part of the current best experimental multi-card setting, while retaining the env gate for fast rollback.
8. Keep the SYCL-aware Meta all-reduce hook for tensor split as the current best experimental path:
   - 2-way direct copy+sum improves the 512-token result from `37.690` to `38.621 tok/s`;
   - 2-way direct single-kernel peer reduction improves the 512-token result again to `39.849 tok/s`;
   - generalized 3-way direct single-kernel peer reduction improves the 512-token result to `41.737 tok/s` after root-order tuning;
   - 4-way direct single-kernel peer reduction reaches `31.482 tok/s`, which is improved but still not production-worthy for single-session speed;
   - 4-way pairwise/hierarchical all-reduce was tested on 2026-05-04 and was slower than the current root single-kernel path (`29.779 tok/s` vs `30.647 tok/s` at 128 generated tokens);
   - next all-reduce work should reduce cross-card reduction frequency or fuse reductions, not just change the reduction copy schedule.
9. Continue the Qwen recurrent split planning work without weakening the Meta scheduler assertion:
   - equal three-way split is now valid;
   - revisit asymmetric two-way splits as a possible balance tool;
   - keep four-way tensor split as a profiling target, not a production path, until communication overhead is reduced.
10. For Vulkan tensor split, profile synchronization and per-op ownership instead of sweeping blind flags.
11. Revisit speculative/draft batching after the non-speculative single-card/multi-card backend reaches the Windows range.
12. Single-card follow-up results:
   - all four B70 selectors land in the same `24.36-24.54 tok/s` range, so do not chase per-slot/card variance;
   - `GGML_SYCL_MMV_Y=2` is slightly worse than default, so keep the default `MMV_Y=1`;
   - forcing DMMV is much slower than reordered MMVQ, so continue focusing on MMVQ/kernel dispatch rather than switching algorithms wholesale.

### Track D: OpenVINO R&D

Goal: determine whether OpenVINO can become a quality-preserving GGUF backend for Qwen3.6.

Steps:

1. Resolve the `GatedDeltaNet` doc/source mismatch:
   - check if the op exists in a newer OpenVINO branch;
   - check if it is plugin-private or generated outside the visible source tree;
   - check whether OpenVINO GenAI packages expose it.
2. If an implementation exists, write a llama.cpp OpenVINO translator for `GGML_OP_GATED_DELTA_NET` targeting it.
3. If not, keep OpenVINO work focused on reducing graph split count but do not expect near-term 27 tok/s.

### Track E: FP8 / Non-GGUF Evaluation

Goal: test whether the official `Qwen/Qwen3.6-27B-FP8` path performs better than GGUF Q4_0 on B70 without unacceptable quality loss.

Context:

- Qwen publishes `Qwen/Qwen3.6-27B-FP8` in Hugging Face Transformers format.
- The model card describes fine-grained FP8 quantization with block size 128 and claims metrics are nearly identical to the original model.
- The model card recommends vLLM, SGLang, KTransformers, and Transformers serving paths, with vLLM/SGLang examples including tensor parallel and MTP/speculative decoding.
- This is not GGUF, so llama.cpp Q4_0 numbers are not directly apples-to-apples. Treat FP8 as a separate backend track.

Plan:

1. Download the official FP8 model:
   - `Qwen/Qwen3.6-27B-FP8`.
   - Keep it separate from GGUF under `/home/steve/models/qwen3.6-27b-fp8-hf`.
2. First test vLLM on Intel/XPU:
   - verify current vLLM Intel/XPU support and required IPEX/oneAPI/PyTorch versions;
   - use existing env `/home/steve/.venvs/vllm-xpu-managed` first (`torch 2.11.0+xpu`, `vllm 0.20.1`, `transformers 5.7.0`);
   - start with 1 GPU, short context, greedy decode, no MTP;
   - then test TP2 and TP4 if the backend supports XPU tensor parallel cleanly.
3. Test SGLang if vLLM/XPU is blocked or slower:
   - start with no speculative decode;
   - then test Qwen3.6 NEXTN/MTP settings from the model card.
4. Keep quality and speed dimensions separate:
   - compare model answers on a small fixed prompt set against the GGUF Q4_0 path;
   - record whether FP8 changes thinking mode behavior or tool-call formatting;
   - benchmark decode tokens/sec with the same prompt/output shapes used for LocalMaxxing.
5. Memory expectations:
   - FP8 weights should be much larger than Q4_0, likely near the practical limit of a single 32 GB B70 once runtime/KV overhead is included;
   - prioritize TP2/TP3 if single-card memory is tight;
   - keep context short first, then scale context only after decode speed is known.
6. Submit comparable results:
   - submit FP8 as `quantization=fp8`;
   - use `engineName=vllm` or `sglang`, not `llama.cpp`;
   - include `tensorParallel`, `specDecoding`, `specMethod`, and exact command snippet.
7. Decision criteria:
   - if FP8 single-GPU decode beats Q4_0 single-GPU by at least 15% with acceptable quality, keep it as a serious path;
   - if FP8 TP2/TP3 beats the current `41.737 tok/s` Q4_0 TP3 result, prioritize FP8 backend optimization;
   - if FP8 is slower but quality is clearly better, keep it as a high-quality mode and continue Q4_0 for speed;
   - if XPU backend support is immature or unstable, document blockers and return to GGUF/SYCL allreduce fusion.

Latest FP8 findings from 2026-05-04:

- Official FP8 download is complete at `/home/steve/models/qwen3.6-27b-fp8-hf`.
- The model is 128x128 block-scaled FP8 Safetensors, not GGUF.
- vLLM/XPU `0.20.1` did not register a block-FP8 XPU kernel for this quantization, so local vLLM now has two experimental XPU paths:
  - quality-preserving BF16 fallback after load-time block-FP8 dequantization;
  - opt-in per-channel FP8 requant path behind `VLLM_XPU_BLOCK_FP8_REQUANT=1`.
- Results are not competitive:
  - TP1 requant: `14.401 s` for 512 prompt + 32 output, about `2.22 output tok/s`;
  - TP4 BF16 fallback: `9.870 s` for 512 prompt + 32 output, about `3.24 output tok/s`;
  - TP4 requant with fixed 2G KV cache: `9.733 s` for 512 prompt + 32 output, about `3.29 output tok/s`;
  - TP2 requant initially OOMed during generation under automatic/profiling cache behavior, but fixed KV sizing avoids that failure;
  - TP2 requant with fixed 1G KV cache: `25.464 s` for 512 prompt + 512 output, output-token upper bound about `20.1 tok/s`;
  - TP4 requant with fixed 1G KV cache: `26.606 s` for 512 prompt + 512 output, output-token upper bound about `19.2 tok/s`.
- FP8 decision:
  - Do not prioritize current vLLM FP8 for speed until there is a native XPU 128x128 block-FP8 W8A8 GEMM path or a better SGLang/KTransformers backend on B70.
  - Preserve the vLLM patch and benchmark wrapper as reproducibility artifacts.
  - Return the primary optimization effort to Q4_0 GGUF SYCL tensor/allreduce fusion and single-card kernel/runtime profiling.
- Artifact clarification:
  - Official FP8 is already local at `/home/steve/models/qwen3.6-27b-fp8-hf`.
  - I have not found a native FP8 GGUF for Qwen3.6 27B. Public GGUF options checked so far are Q4/Q6/Q8/BF16-style GGUFs, or GGUFs derived from an FP8 source but quantized to a non-FP8 GGUF type.
  - A Q8_0 GGUF fallback download is in progress under `/home/steve/models/qwen3.6-27b-q8_0-gguf`. This is useful for quality-preserving GGUF comparison but should not be reported as FP8.

### Track F: Current Next Software Targets

Goal: improve quality-preserving Q4_0 performance without power-limit changes.

1. Single-card SYCL profiling:
   - profile the reordered Q4_0 MMVQ decode path and recurrent kernels rather than continuing broad flag sweeps;
   - focus on `reorder_mul_mat_vec_q4_0_q8_1_sycl`, `quantize_row_q8_1_sycl`, `GATED_DELTA_NET`, `SSM_CONV`, and recurrent state copies;
   - compare Level Zero queue/synchronization behavior against the Windows SYCL command that reaches `>27 tok/s`.
2. Multi-card Q4_0:
   - keep the 3-GPU single-kernel allreduce path as the current best speed path (`41.737 tok/s`);
   - stop spending time on 4-GPU reduction topology sweeps until the number of reductions per token is reduced;
   - investigate fusing or delaying Meta allreduces across safe graph regions so 4-GPU does not pay 128 tiny cross-card reductions per token.
   - 4-GPU allreduce-order trace confirms the reductions are dependency-ordered as two reductions per layer (`attn/linear_attn_out-N` then `ffn_out-N`), all `20480` bytes; simple adjacent packing is not enough.
3. Alternative multi-GPU decomposition:
   - revisit row split only if activation broadcast/gather waits can be removed or fused;
   - evaluate whether pipeline/layer partitioning can help multi-session throughput, but do not count it as single-session speed unless decode overlap is implemented.
4. Backend exploration:
   - keep OpenVINO as R&D around Qwen recurrent ops and graph fragmentation;
   - treat FP8 as blocked on native XPU block-FP8 kernels or a better backend;
   - SGLang/KTransformers FP8 can be explored later, but should not displace the current Q4_0 work unless it shows a credible path above `41.737 tok/s`;
   - add Intel `llm-scaler` as an explicit idea/experiment source. It may not map cleanly to Arc/B70, but inspect its XPU sharding, communication, serving, and kernel choices for patterns we can adapt.
5. Large-model follow-on:
   - after the 27B path is exhausted or no longer the best use of time, test a true capacity run across all four B70s;
   - target: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS`;
   - local files total about `101G`, so it should barely fit in 128GB aggregate VRAM if tensor split overhead and KV cache are controlled;
   - start with minimum context and one-token load/sanity before any speed claims.

### Track G: Immediate Follow-Up Queue

1. Finish the current llama.cpp rebuild with stderr Meta allreduce stats.
2. Run a short 3-GPU tensor-split decode trace with `GGML_META_ALLREDUCE_STATS=2` and the current best selector order `2,1,3`. Done: trace confirms 128 reductions/token, all `20480` bytes, one attention-output and one FFN-output reduction per layer.
3. Use the allreduce tensor-name trace to classify reductions:
   - mandatory synchronization before nonlinear/recurrent state updates;
   - reductions that might be safely delayed through elementwise or normalization-adjacent ops;
   - repeated projection reductions that could be batched into one larger communication unit.
4. Keep single-card Q4_0 work focused on `MUL_MAT` / reordered MMVQ because profiling shows it dominates decode time. Lower-priority recurrent kernels are already under 1% each in the one-token profile.
5. Do not spend more time on simple launch-constant sweeps unless a profiler points at a specific occupancy/register issue:
   - `GGML_SYCL_MMV_Y=2` was slower;
   - `GGML_SYCL_REORDER_MMVQ_SUBGROUPS=8` was slower;
   - forced DMMV was much slower.
6. After the Q8_0 GGUF finishes downloading, run only short memory/sanity tests first. If it fits on 2-4 B70s, use it as a high-quality GGUF comparison mode, not as a replacement for the Q4_0 speed target.
7. New immediate measurement: run `GGML_META_ALLREDUCE_STATS=3` on 3-GPU and 4-GPU short decode after the timing-hook rebuild finishes. Use the synchronized timing only for diagnosis because it intentionally disturbs normal scheduling.
   - Done for 2/3/4 GPUs. Steady synchronized allreduce cost per token is roughly `1.718 ms`, `5.732 ms`, and `10.605 ms` respectively for the same 128 20 KiB reductions.
   - This confirms the 4-GPU regression is communication/synchronization overhead, not bad root ordering or an obvious tensor-size issue.
8. Q8_0 GGUF fallback is downloaded. Run short fit/sanity tests:
   - first single GPU with one generated token;
   - then 2-4 GPUs only if single-card memory is tight or if the high-quality GGUF path is worth comparing.
   - Done:
     - 1x B70 Q8_0, 512 prompt / 128 output: `15.275 tok/s` decode;
     - 2x B70 Q8_0 tensor split, 512 prompt / 128 output: `25.733 tok/s` decode;
     - 3x/4x B70 Q8_0 tensor split aborts in model tensor allocation with `GGML_ASSERT(buffer)`.
   - Decision: Q8_0 is a high-quality 1-2 GPU mode, but Q4_0 remains the speed target. Add Q8_0 TP3/TP4 allocator failure to the bug queue rather than optimizing Q8_0 first.
9. Single-card Q4_0 graph check:
   - `GGML_SYCL_DISABLE_GRAPH=0`: `24.427 tok/s`;
   - `GGML_SYCL_DISABLE_GRAPH=1`: `24.343 tok/s`;
   - decision: graph capture is not the current single-card bottleneck.
10. Single-card Q4_0 reorder check:
   - `GGML_SYCL_DISABLE_OPT=0`: `24.356 tok/s`;
   - `GGML_SYCL_DISABLE_OPT=1`: `15.227 tok/s`;
   - decision: the reordered Q4_0 MMVQ path is required. Optimize within it; do not fall back to plain MMVQ.
11. Single-card Q4_0 launch/runtime checks:
   - `VDR_Q4_0_Q8_1_MMVQ=4`: `24.431 tok/s`, neutral versus control;
   - best `-fa`/`-ub` sweep point: `24.406 tok/s` with `-fa 0 -ub 64`;
   - decision: flash attention, ubatch, graph capture, and simple Q4_0 vector-dot width changes do not explain the Linux single-card gap. The next single-card step must inspect the reordered MMVQ kernel/dataflow itself.
12. 4-GPU allreduce topology follow-up:
   - added env-gated `GGML_SYCL_COMM_ROOT_COPY=1`;
   - result: 2x `38.259 tok/s`, 3x `39.817 tok/s`, 4x `30.371 tok/s`;
   - decision: root-copy is stable but slower than single-kernel allreduce. Keep it only as a diagnostic branch. The 4-GPU problem requires fewer/smarter synchronization points, not another simple root topology.
13. FP8 retest with manual KV sizing:
   - TP2 fixed-KV FP8: 512 input / 512 output in `25.464 s`;
   - TP4 fixed-KV FP8: 512 input / 512 output in `26.606 s`;
   - decision: official FP8 is runnable and useful to preserve, but not currently a speed path on B70 without native block-FP8 XPU kernels.
14. FP8 artifact clarification:
   - official local FP8 is already complete at `/home/steve/models/qwen3.6-27b-fp8-hf`;
   - it is dynamic E4M3 block-FP8 HF/Safetensors with 128x128 weight blocks, not GGUF;
   - current public "FP8 GGUF" hits are Q4_K_M GGUF conversions from an FP8 source, not native FP8 GGUF;
   - decision: do not download duplicate HF/Safetensors FP8 variants unless a backend comparison needs a different revision.
15. Q4_0 `MUL_MAT` stage profiling:
   - added `GGML_SYCL_MUL_MAT_STATS=1` to split explicit-sync time into activation quantization, peer copies, and matvec kernels;
   - single-card one-token profiles show hundreds of Q8_1 activation quantize launches per generated token before reordered MMVQ kernels;
   - forced DMMV is still a throughput loss (`15.518 tok/s` over 64 generated tokens versus `22.683 tok/s` same-shape MMVQ control);
   - selective `result_output` DMMV is also a loss (`24.007 tok/s` over 256 generated tokens versus `24.426 tok/s` same-build MMVQ control);
   - decision: keep DMMV env switches diagnostic-only. Next useful work is reducing activation-quantize launch overhead or improving graph scheduling around the current reordered MMVQ path.
16. Q8_1 activation-cache prototype:
   - added `GGML_SYCL_Q8_CACHE=1`, a graph-scoped cache for Q8_1 activations used by Q4_0 `MUL_MAT`;
   - single-card result is neutral: `24.425 tok/s` off, `24.500 tok/s` on for 512 prompt / 256 output;
   - 2x B70 selector `0,3`: `40.083 tok/s` off, `40.684 tok/s` on for 512 prompt / 256 output; longer 512-output validation with cache on: `40.487 tok/s`;
   - 3x B70 selector `2,1,3`: `40.937 tok/s` off, `42.432 tok/s` on for 512 prompt / 256 output; longer 512-output validation with cache on: `41.659 tok/s`;
   - 4x B70 selector `0,1,2,3`: `31.913 tok/s` with cache on for 512 prompt / 128 output;
   - decision: keep env-gated and submit/document as a modest multi-GPU software optimization. It preserves quality because it reuses the exact same Q8_1 activation within one graph compute. The 4-GPU result remains bottlenecked by allreduce/synchronization, not activation quantization.
   - LocalMaxxing follow-up submissions:
     - 2x B70 512-output validation `40.487 tok/s`: `cmormylxz000fib04wodwo1ng`;
     - 3x B70 512-output validation `41.659 tok/s`: `cmorn71e2000kib0415vo51vj`.
     - 4x B70 Q8-cache negative-scaling diagnostic `31.913 tok/s`: `cmornec37000okw040zl9563z`.
   - 4-GPU one-token allreduce trace with cache enabled:
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-allreduce-order-quad0123-q8cache-p0n1-20260504T164916Z.log`;
     - each pass reports `128` reductions, all `20480` bytes, ordered as `linear_attn_out-N` / `attn_output-N` then `ffn_out-N` for all 64 layers;
     - decision: do not prototype blind adjacent-packing first. Prototype delayed/fused reduction through safe linear/residual regions or fused row-parallel output kernels.
17. MTP/speculative decode on vLLM/XPU:
   - vLLM has Qwen MTP wiring and Qwen3.6 FP8 includes MTP weights;
   - current XPU Gated DeltaNet path asserts when speculative sequence masks are present (`vllm/_xpu_ops.py`);
   - next MTP experiment is a correctness-only fallback patch in `GatedDeltaNetAttention.forward_xpu`, not a priority speed path until XPU speculative GDN/Triton support is proven.
18. FP8 artifact/download status:
   - official dynamic E4M3 block-FP8 HF/Safetensors is complete locally;
   - no native Qwen3.6 FP8 GGUF was found; public "FP8 GGUF" hits are Q4_K_M GGUF conversions from an FP8 source;
   - static/tensor FP8 variant `vrfai/Qwen3.6-27B-FP8` is being downloaded via resumable `curl` because `hf download` stalled on the large file.
19. `llm-scaler` review:
   - repo exists locally at `/home/steve/src/llm-scaler` (`origin=https://github.com/intel/llm-scaler.git`);
   - review for B70-relevant ideas: XPU process/device placement, tensor-parallel communication, KV/cache layout, speculative/decode scheduling, and any Intel-specific environment or kernel flags;
   - do not assume it will run well on Arc/B70. Treat it first as a reference implementation and only run it if setup cost is low.
   - first review result:
     - current `origin/main` is `e0b0703` from `2026-04-29`;
     - useful references found: XPU `reduce_scatter` / `all_gatherv`, all-gather/reduce-scatter expert communication, `SKIP_ALL_REDUCE` diagnostic, fused norm+GEMV INT4/FP8 kernels, QKV split+norm+RoPE, Gated DeltaNet/conv decode kernels, EAGLE/MTP kernels, oneDNN FP8 primitive caching, and GGUF batch dequantization;
     - strongest llama.cpp follow-up is still fewer/fused reductions for 4-GPU decode, now informed by their reduce-scatter/all-gather and fused-output-kernel direction.
20. MiniMax M2.7 4-GPU capacity test:
   - path: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS`;
   - files present: four GGUF shards totaling about `101G`;
   - tensor split smoke confirmed the current llama.cpp tree does not implement `LLAMA_SPLIT_MODE_TENSOR` for `minimax-m2`;
   - layer split loads metadata but fails on large contiguous SYCL buffers even at `-ngl 32` (`13.8G` allocation failure);
   - row split is closer, but fails on split expert tensor allocation (`160432128` bytes, consistent with one `ffn_down_exps` shard);
   - rebuilt with allocation diagnostics; first confirmed row-split failure is `blk.12.ffn_down_exps.weight`, type `iq4_xs`, device `1`, rows `[196608, 393216)`, allocation `160432128` bytes;
   - fallback row split with `-ncmoe 62` loads, constructs context, and reserves the graph, proving shard handling works, but all experts sit on CPU/file-backed memory and first-token speed is not useful on this 15G RAM machine;
   - next action: patch or override the split expert tensor path, starting with `ffn_down_exps`/`iq4_xs`; do not submit MiniMax performance numbers until expert tensors run on GPU.
21. Static FP8 vLLM/XPU branch:
   - `vrfai/Qwen3.6-27B-FP8` is downloaded at `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
   - TP2 is not viable on the current memory layout: OOM around LM-head allocation;
   - TP4 with default XPU FlashAttention originally loaded, then failed first forward on Intel XPU FA2 `k_descale` scalar assertion;
   - root cause: the checkpoint's compressed-tensors attention scales are singleton `(1,)` tensors, while Intel's XPU FA2 wrapper requires scalar-view descale tensors;
   - local patch applied in source and active venv:
     - `/home/steve/src/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`;
     - `/home/steve/.venvs/vllm-xpu-managed/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`;
     - reshape singleton attention scales with `tensor.reshape(())` before storing `_q_scale/_k_scale/_v_scale`;
     - quality impact from the patch itself: none expected, because scale values are unchanged.
   - TP4 with `--attention-backend TRITON_ATTN` runs successfully:
     - 512 prompt / 256 output, 5 measured iters, avg latency `7.7385800618 s`;
     - computed output throughput `33.081 tok/s`;
     - LocalMaxxing submission `cmorijmiu000kjr04cour40px`;
   - TP4 with patched XPU FA2 now runs successfully:
     - 512 prompt / 256 output, 1 warmup + 5 measured iters, avg latency `6.5199336920 s`;
     - computed output throughput `39.264 tok/s`;
     - computed total throughput `117.793 tok/s`;
     - LocalMaxxing submission `cmorjwgi8000el7041jdd8faa`;
     - 512 prompt / 512 output, 1 warmup + 5 measured iters, avg latency `12.3364390012 s`;
     - computed output throughput `41.503 tok/s`;
     - computed total throughput `83.006 tok/s`;
     - LocalMaxxing submission `cmork3n3k000ujo04y73lbr1j`;
     - this longer-output result is the best FP8 sustained decode run so far and is effectively tied with the current Q4_0 TP3 validation.
   - CCL screen:
     - `ofi` + default `pidfd` was marginally faster on a two-iteration screen but not on five-iteration validation;
     - `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` did not help;
     - keep topology override disabled unless a later allreduce-specific test proves it helps.
   - XPU graph screen:
     - PyTorch reports XPU graph support, but vLLM disables cudagraph mode under TP4 because communication ops are present;
     - `VLLM_XPU_ENABLE_XPU_GRAPH=1` therefore does not change steady-state speed meaningfully (`6.513733 s` avg for 512/256 on a two-iteration screen);
     - do not spend more time on XPU graph for TP4 until vLLM can capture or isolate TP communication.
   - decision: patched FA2 is now the best FP8 path. It does not beat the Q4_0 TP3 peak yet, but it is a major FP8 improvement over Triton and preserves more model fidelity than INT4 AutoRound.
22. Static FP8 full-context / 2x2 topology follow-up:
   - added a language-only vLLM patch for `Qwen3_5ForConditionalGeneration` so `--language-model-only` avoids constructing the unused vision tower and skips `visual.*` weights;
   - plain TP2 still fails around `lm_head` allocation, so two-card FP8 is not viable yet on the current vLLM/XPU memory path;
   - PP2 x TP2 across four B70s is valid with default `pidfd` IPC:
     - sockets IPC fails when pipeline point-to-point communication starts;
     - 512 prompt / 256 output, 1 warmup + 2 measured iterations: `22.721 tok/s` output;
     - LocalMaxxing diagnostic submission: `cmormmlz0000bky04wpu4oc01`;
     - `--max-model-len 262144` succeeds and reports `1,138,148` GPU KV-cache tokens / `4.34x` concurrency for 262k-token requests.
   - TP4 also succeeds at `--max-model-len 262144`:
     - reports `1,206,355` GPU KV-cache tokens / `4.60x` concurrency for 262k-token requests;
     - cold tiny 32/8 smoke is faster than PP2 x TP2 (`11.321 s` vs `18.293 s`);
     - therefore TP4 remains the preferred four-card Qwen3.6 FP8 topology for both speed and context.
   - FP8 KV cache was tested and is not a speed path:
     - 512 prompt / 256 output with `--kv-cache-dtype fp8`: `28.036 tok/s`;
     - LocalMaxxing diagnostic negative submission: `cmornlh8g000vkw04yb57ukvl`;
     - vLLM warns of possible accuracy drop without proper scaling;
     - keep auto/BF16 KV for quality-preserving speed runs.
   - decision: keep PP2 x TP2 as a capacity fallback for larger models, but optimize Qwen3.6 27B FP8 on TP4.
23. Q4_0 rotate-root allreduce diagnostic:
   - added env-gated `GGML_SYCL_COMM_ROTATE_ROOT=1` for the single-kernel allreduce path;
   - default behavior remains unchanged when the env var is unset;
   - rebuild note: final linking needs oneAPI environment sourced with `source /opt/intel/oneapi/setvars.sh --force`;
   - 4x B70 short smoke, 512 prompt / 32 output:
     - rotate off: `30.494 tok/s`;
     - rotate on: `30.729 tok/s`;
   - 3x B70 selector `2,1,3`, 512 prompt / 128 output:
     - rotate off: `42.573 tok/s`;
     - rotate on: `41.359 tok/s`;
   - 4x B70 selector `0,1,2,3`, 512 prompt / 128 output:
     - rotate off: `32.165 tok/s`;
     - rotate on: `31.754 tok/s`;
   - decision: root rotation is stable but not a speed path. Keep it diagnostic-only and do not submit it as an improvement. The four-card problem is still the count and cost of 128 small reductions/token.
   - allreduce trace interpretation: reduction tensors are direct `MUL_MAT` outputs (`linear_attn_out` / `attn_output` and `ffn_out`), not cheap metadata-only view/reshape boundaries, so the next useful Q4_0 implementation work is fused matmul/allreduce epilogues, reduce-scatter/all-gather-style decomposition, or a lower-overhead tiny reduction primitive.
24. MiniMax M2.7 source diagnosis follow-up:
   - source review confirms `LLAMA_SPLIT_MODE_TENSOR` is intentionally unsupported for `minimax-m2`;
   - row split reaches the intended SYCL split-buffer path and fails first on `blk.12.ffn_down_exps.weight`, type `iq4_xs`, rows `[196608, 393216)`, allocation `160432128` bytes;
   - the failed allocation matches the expected row-slice size for the expert tensor, so the failure is in split-buffer allocation/capacity behavior rather than bad range math;
   - even after allocation is fixed, `GGML_OP_MUL_MAT_ID` with SYCL split buffers is not implemented and is likely to assert in execution;
   - four-B70 row-split `-ngl 11`, `-p 0 -n 1` confirmation timed out after 240 seconds with empty output, so no MiniMax performance number is valid yet;
   - next viable patches:
     - low-risk diagnostic: keep `MUL_MAT_ID` expert tensors off SYCL split buffers until split execution is implemented;
     - medium-risk diagnostic: add host-USM fallback for failed split-buffer tensor allocations;
     - real speed path: implement expert-aligned split `MUL_MAT_ID`, where selected experts run only on the owning B70 and outputs are assembled correctly.
25. vLLM/XPU speculative decode follow-up:
   - fixed the local FP8 benchmark wrapper so the static FP8 VRFAI checkpoint can be run with `QUANTIZATION=compressed-tensors` instead of the wrapper always forcing `--quantization fp8`;
   - TP4 MTP smoke reaches target/draft model resolution and XCCL rank initialization, then hangs before useful generation; a short `strace` sample showed rank 1 spinning in `sched_yield`, so the next MTP work is startup/synchronization diagnosis rather than decode-kernel tuning;
   - TP4 n-gram speculative decode originally failed in the XPU Gated DeltaNet custom op because `non_spec_state_indices_tensor` was not contiguous;
   - patched source and active venv `_xpu_ops.py` to pass contiguous `non_spec_state_indices_tensor` and `non_spec_query_start_loc` into `_xpu_C.gdn_attention`;
   - mirrored the XPU Gated DeltaNet speculative fallback in source so speculative sequence masks can route through the generic `_forward_core` path instead of the fused XPU op that asserts on speculative masks;
   - rerun result: n-gram speculative decode now completes a cold TP4 `32/8` smoke, `avg_latency=6.131081808 s`;
   - controlled n-gram screen, TP4 static FP8, `512/256`: `42.245 tok/s` output versus about `39.264 tok/s` same-shape TP4 FP8 FA2 baseline;
   - controlled n-gram validation, TP4 static FP8, `512/512`: `42.489 tok/s` output versus `41.503 tok/s` same-shape TP4 FP8 FA2 baseline;
   - LocalMaxxing submission for `512/512`: `cmorr43b30004jj04h4hhb6v1`;
   - `num_speculative_tokens=4` validation, TP4 static FP8, `512/512`: `46.067 tok/s` output;
   - LocalMaxxing submission for `num_speculative_tokens=4`: `cmorre1hq000fi30421gxpv3j`;
   - `num_speculative_tokens=6` stalled during initialization/profile with CPU-hot workers and `No available shared memory broadcast block found in 60 seconds`, then was terminated;
   - `num_speculative_tokens=5` also stalled before useful output, with rank 0 CPU-hot and the other workers mostly idle;
   - lookup min/max `2/7` with `4` draft tokens looked good on a two-iteration screen (`49.133 tok/s`) but validated lower at `45.285 tok/s`;
   - decision: n-gram speculative decode is now a real quality-preserving software speed path for the static FP8 checkpoint, but current best remains `4` draft tokens with lookup min/max `2/5`.
26. MTP speculative decode next boundary:
   - read-only review points to startup/draft TP synchronization rather than the already-patched n-gram GDN metadata issue;
   - primary files/functions to instrument next:
     - `vllm/v1/worker/xpu_worker.py`, `XPUWorker.init_device`, especially distributed init and the XPU all-reduce warmup;
     - `vllm/v1/worker/gpu_model_runner.py`, speculative setup and drafter creation;
     - `vllm/v1/spec_decode/draft_model.py`, draft TP mismatch handling;
     - `vllm/model_executor/models/qwen3_5_mtp.py`, `Qwen3_5MultiTokenPredictor` TP collectives;
     - `vllm/v1/worker/xpu_model_runner.py`, CUDA compatibility wrapper for XPU events/streams;
   - low-risk diagnostics: TP1 MTP compatibility smoke, `--enforce-eager`, MTP JSON `enforce_eager`, and `disable_padded_drafter_batch`;
   - do not run heavy MTP benchmarks until the TP1 or eager smoke proves the startup path is not deadlocking.
27. Q4_0 allreduce next boundary:
   - read-only review confirmed the current 4-GPU bottleneck is many tiny reductions: `128` reductions/token, each `20480` bytes, contiguous F32 direct `MUL_MAT` outputs;
   - best first patch boundary is an env-gated small-F32 allreduce fast path inside `ggml_backend_sycl_comm_allreduce_tensor()` for `nbytes == 20480`, `type == GGML_TYPE_F32`, and `n_backends in {2,3}` before changing meta graph semantics;
   - larger fused MMVQ/allreduce epilogue remains the likely ceiling, but it crosses `ggml_sycl_op_mul_mat()`, `ggml_sycl_op_mul_mat_vec_q()`, and meta backend scheduling, so quantify the small-reduction overhead floor first.
28. Q4_0 small-F32 allreduce diagnostic result:
   - added env-gated `GGML_SYCL_COMM_SMALL_F32=1` for contiguous F32 `20480` byte allreduce tensors on `2` or `3` SYCL backends;
   - implementation used one fixed `256` work-item kernel and dependency barriers on non-root queues;
   - 3x B70 selector `2,1,3`, `512/128` A/B:
     - control: `42.773 tok/s`;
     - small-F32 diagnostic: `34.985 tok/s`;
   - decision: this specific small-kernel shape is not useful. It underutilizes the GPU despite the tiny tensor size. If we continue this route, test only the barrier/event change while keeping full-range parallelism, or move to the larger fused MMVQ/DMMV allreduce epilogue.
29. MiniMax M2.7 row-split `-ncmoe` staircase:
   - four-GPU row split with `-ncmoe 13`, `26`, `38`, and `50` all failed before generation, but the failure moved predictably later as more expert layers were forced to host:
     - `-ncmoe 13`: first failed GPU expert allocation at `blk.25.ffn_gate_exps.weight`, `129761280` bytes on device `1`;
     - `-ncmoe 26`: first failed GPU expert allocation at `blk.37.ffn_up_exps.weight`, `129761280` bytes on device `0`;
     - `-ncmoe 38`: first failed GPU expert allocation at `blk.49.ffn_gate_exps.weight`, `129761280` bytes on device `1`;
     - `-ncmoe 50`: first failed GPU expert allocation at `blk.60.ffn_up_exps.weight`, `129761280` bytes on device `0`;
   - interpretation: each B70 is only absorbing about 12 to 13 GPU-resident expert layers in the current SYCL row-split allocation path before even a small split expert slice fails;
   - `-ncmoe 62` remains a proof that shard loading and context construction can work, but it is CPU/file-backed for all experts and is not a useful performance path on this low-RAM host;
   - next action: stop treating MiniMax as a flag-tuning problem. The required code work is split expert allocation plus `GGML_OP_MUL_MAT_ID` execution on SYCL split buffers, or an expert-aligned alternative that keeps selected experts on the owning B70.
30. Q4_0 event-barrier allreduce diagnostic:
   - added env-gated `GGML_SYCL_COMM_EVENT_BARRIER=1` for the existing experimental single-kernel allreduce path;
   - change: non-root queues use `ext_oneapi_submit_barrier({reduce})` after the root allreduce kernel instead of submitting a tiny dependent `single_task`;
   - 3x B70 selector `2,1,3`, `512/128` A/B:
     - event barrier off: `41.983 tok/s`;
     - event barrier on: `43.331 tok/s`;
   - 3x B70 selector `2,1,3`, `512/512`, `3` repeats:
     - samples: `43.9488`, `43.4813`, `43.385 tok/s`;
     - average: `43.605 tok/s`;
     - LocalMaxxing: `cmortp5vn000el404dj3zqv0u`;
   - interpretation: replacing marker tasks with event barriers is a real quality-preserving synchronization win on the current TP3 path. It does not change math, model weights, KV dtype, sampling, or power limits.
   - 2x follow-up, selector `0,3`, `512/256`: event barrier off `40.457 tok/s`, event barrier on `40.331 tok/s`; no improvement;
   - 4x follow-up, selector `0,1,2,3`, `512/128`: event barrier off `31.910 tok/s`, event barrier on `32.427 tok/s`; slight improvement but still far below 2x/3x;
   - next action: stop changing allreduce launch markers for 4x. The remaining 4x bottleneck is reduction frequency/fanout, so move to fused matmul/allreduce epilogues or reduce-scatter-style graph changes.
31. 2026-05-05 follow-up screens:
   - Q4_0 `GGML_SYCL_COMM_SMALL_F32=1` is confirmed negative in the current implementation:
     - 4x selector `0,1,2,3`, `512/128`, event barrier on: `31.763 tok/s`, below the prior `32.427 tok/s`;
     - 3x selector `2,1,3`, `512/128`, event barrier on: `34.874 tok/s`, far below the 3x event-barrier path without small-F32;
     - decision: keep small-F32 disabled. The fixed 256-work-item tiny reduction underutilizes B70 and is not worth more tuning in this shape.
   - FP8 TP2/PP2:
     - no-spec warm-cache `64/64`: `27.795 tok/s`;
     - PP2+n-gram `512/128` hits a vLLM/XPU bug: `XPUModelRunner` has no `drafter`, then workers hang;
     - decision: PP2 is a capacity fallback, not a Qwen3.6 27B single-stream speed path.
   - FP8 TP4 oneCCL topology override:
     - `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, n-gram `512/256`: `40.049 tok/s`;
     - default topology same shape remains better at `42.245 tok/s`;
     - decision: leave oneCCL topology recognition enabled.
   - MiniMax `MUL_MAT_ID` guard:
     - capability masking moves the failure from split expert slices to monolithic SYCL buffer allocation (`26.877 GB` at `-ncmoe 13`, `20.158 GB` at `-ncmoe 50`);
     - decision: masking is diagnostic only. The real implementation target is split-buffer `GGML_OP_MUL_MAT_ID` or an expert-owned execution plan, not fallback placement.
32. Next implementation targets:
   - Q4_0:
     - design a fused `MUL_MAT` output-projection plus small allreduce path for the direct `linear_attn_out` / `attn_output` 20 KB F32 outputs;
     - alternatively prototype a reduce-scatter/all-gather-like meta scheduling path that avoids writing a fully mirrored 20 KB tensor on every B70 for every reduction;
     - use 3x event-barrier `43.605 tok/s` as the regression guard and 4x event-barrier `32.427 tok/s` as the main target to beat.
   - FP8:
     - keep TP4/default oneCCL/n-gram4 lookup `2/5` as current best;
     - investigate the PP2+n-gram `drafter` attribute bug only if PP2 becomes necessary for a larger model.
   - MiniMax:
     - add loader-placement diagnostics for `blk.*.ffn_*_exps.weight` buffer type choices;
     - implement or prototype split-buffer `MUL_MAT_ID` handling before more `-ncmoe` sweeps.
33. MiniMax split `MUL_MAT_ID` token-generation prototype:
   - added env-gated `GGML_SYCL_MUL_MAT_ID_SPLIT=1` in the SYCL backend;
   - first version targets token generation only (`ne12 == 1`) and maps selected expert IDs to split-buffer row shards across the four B70s;
   - patch artifact: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-minimax-split-mulmatid-tg.patch`;
   - `-ncmoe 50` no longer falls back to a 20 GB monolithic SYCL3 expert allocation, confirming the capability/placement blocker was removed;
   - `-ncmoe 60` loaded the model, constructed context, assigned KV over all four B70s, enabled fused Gated Delta Net paths, and reserved a `3975` node / `122` split graph;
   - status: first one-token decode did not complete; after more than eight minutes, `strace` showed the main thread spinning in `sched_yield()` with an empty JSONL result;
   - decision: this is a real implementation lead, not a usable runtime path yet. Next MiniMax step is instrumentation around the first split `MUL_MAT_ID` call plus batched expert-shard execution/copy-path cleanup, not more flag sweeps.
34. Runtime recovery and corrected Q4 screen:
   - MiniMax stall produced an `xe` coredump/reset on `0000:03:00.0`; the first post-MiniMax Q4 run produced the same `sched_yield()` spin and an `xe` schedule-disable failure/reset on `0000:83:00.0`;
   - with no active compute processes, debugfs GT resets restored runtime health and a tiny single-B70 Q4 smoke succeeded;
   - corrected DNN-off validation:
     - 3x selector `2,1,3`, `512/128`: `42.170 tok/s` decode;
     - 4x selector `0,1,2,3`, `512/128`: `32.477 tok/s` decode;
   - decision: the 4x Q4 regression is confirmed under the correct DNN-off stack. Do not spend more time on basic selector/flag sweeps until a fusion/reduced-collective implementation exists.
35. FP8 TP4 validation blocker:
   - best unsubmitted candidate remains static FP8 TP4 n-gram, `num_speculative_tokens=4`, lookup min/max `2/4`, `512/512`, two measured iterations at `50.193 tok/s`;
   - attempted a `3` iteration validation after the GPU reset, but vLLM failed before benchmark JSON with a segfault in XCCL communicator initialization, entering oneCCL `coll_init` and SYCL/UR Level Zero `urProgramBuildExp`;
   - standalone XCCL allreduce now segfaults even at `2` ranks after stale `/dev/shm` cleanup, NEO compiler cache removal, topology override, MPI transport attempt, and GT resets;
   - decision: do not submit the `50.193 tok/s` candidate yet and do not run more TP vLLM benchmarks in this boot session. Reboot or driver reload is needed before the next FP8 validation.
36. 2026-05-05 follow-up implementation screens:
   - MiniMax split `MUL_MAT_ID` host-bounce copy path was added behind `GGML_SYCL_MUL_MAT_ID_SPLIT_HOST_BOUNCE=1`, but a four-B70 `-ncmoe 60`, one-token attempt still timed out and triggered new `xe` CCS/BCS engine resets. Next MiniMax work needs a synthetic split-expert harness or flushed per-stage heartbeat instrumentation before another full-model decode attempt.
   - Q4_0 `GGML_SYCL_COMM_SKIP_ROOT_READY=1` was neutral on 4x B70: same-build control `32.720377 tok/s`, skip-root `32.776781 tok/s` for 512 prompt / 128 output. Keep it diagnostic-only.
   - standalone two-rank XCCL remains unhealthy after these screens, with both ranks segfaulting before measured allreduce rows. FP8 TP validation stays paused until reboot or driver reload.
   - artifacts:
     - `/home/steve/llm-optimization-artifacts/notes/2026-05-05-followups-minimax-q4-xccl.md`;
     - `/home/steve/llm-optimization-artifacts/data/followups-minimax-q4-xccl-20260505.json`;
     - `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-followups-minimax-hostbounce-q4-skiprootready-20260505.patch`;
     - `/home/steve/llm-optimization-artifacts/patches/vllm-xpu-qwen36-fp8-fa2-ngram-language-only-20260505.patch`.
37. 2026-05-05 Q4_0 fused allreduce + residual add:
   - added a new backend proc, `ggml_backend_comm_allreduce_add_tensor`, plus a SYCL implementation for the existing single-kernel allreduce path;
   - added Meta graph gate `GGML_META_FUSE_ALLREDUCE_ADD=1` to detect immediate `ADD(partial, mirrored)` / `ADD(mirrored, partial)` patterns and skip the separate residual ADD subgraph when the backend fused path accepts the tensors;
   - correctness guard: the fused kernel computes `sum(partial shards) + mirrored_residual`, so the residual is not allreduced and the result is mathematically equivalent to the baseline allreduce followed by ADD;
   - dual-card one-token diagnostics:
     - `GGML_META_FUSE_ALLREDUCE_ADD_LIMIT=1` routed exactly one reduction through `path=backend+add`;
     - all-site mode routed `79` of `128` per-token reductions through `path=backend+add`;
   - 3x selector `2,1,3`, Qwen3.6 27B Q4_0, 512/128 screen with the fast historical command shape: `44.016108 tok/s`;
   - 3x selector `2,1,3`, 512/512, `3` repeats validation:
     - prompt: `135.665459 tok/s`;
     - decode samples: `43.9985`, `44.0192`, `43.9953 tok/s`;
     - average decode: `44.004344 tok/s`;
     - total throughput: `66.453788 tok/s`;
     - validation JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-fuseadd-validate-triple213-p512n512-r3-20260505T025706Z.jsonl`;
   - LocalMaxxing accepted reduced payload `cmos1jmsv000iih04iifehc8d`;
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-05-q4-fused-allreduce-add.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-fused-allreduce-add-20260505.json`;
     - focused patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-fused-allreduce-add-focused-20260505.patch`;
   - measurement caveat: forcing `-b 512` now lowers 3x controls to about `30.7-30.9 tok/s`; leaving `-b` unset reports `n_batch=2048` and matches the healthy `42+ tok/s` control path. Keep command shape consistent for future comparisons.
   - follow-up 4x selector `0,1,2,3`, no explicit `-b`, 512/128:
     - control: `32.383337 tok/s`;
     - fused-add: `33.219955 tok/s`;
     - conclusion: fused-add helps slightly on 4x but does not solve the major negative scaling.
   - follow-up 2x selector `0,3`, explicit `-b 512`, 512/256:
     - control: `40.278630 tok/s`;
     - fused-add: `40.265194 tok/s`;
     - conclusion: fused-add is neutral on the best dual-card command shape.
   - next action: prototype a 4x collective variant that avoids the current root-kernel remote-write fanout, then investigate whether the remaining 49 non-fused reductions can be safely matched or require a lower-level matmul/allreduce epilogue.
38. 2026-05-05 Q4_0 4x collective negative prototypes:
   - added env-gated `GGML_SYCL_COMM_LOCAL_WRITE=1`:
     - ordinary in-place allreduce gathers all peer partials into per-device temporary buffers, then each GPU writes only its local mirrored output;
     - fused-add sites use local per-device kernels directly because the output tensor is separate from the partial tensor.
   - 4x selector `0,1,2,3`, Qwen3.6 27B Q4_0, 512/128:
     - root fused-add baseline from item 37: `33.219955 tok/s`;
     - `GGML_SYCL_COMM_LOCAL_WRITE=1`: `30.681785 tok/s`;
     - conclusion: per-device temp gather removes root remote writes but adds too many tiny copies/events for decode. It is not a useful 4x path.
   - narrowed `GGML_SYCL_COMM_LOCAL_WRITE=2` to fused-add sites only, leaving ordinary allreduces on the root single-kernel path:
     - result: `32.365769 tok/s`;
     - conclusion: four local fused-add kernels per fused site are still slower than one root fused kernel.
   - added env-gated `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1`:
     - uses the root residual for all outputs at fused-add sites because the residual tensor is mirrored by the Meta split state;
     - result: `33.074515 tok/s`;
   - conclusion: remote residual reads are not the main 4x bottleneck.
   - decision: keep these as diagnostic-only env gates. The remaining 4x speed work likely needs fewer reductions or a true matmul/reduction epilogue, not alternate tiny-copy or tiny-kernel collective topologies.
39. 2026-05-05 FP8/XCCL recovery and validation:
   - standalone XCCL 2-rank gate recovered without a reboot:
     - selector `level_zero:0,1`, `CCL_ZE_IPC_EXCHANGE=sockets`;
     - completed allreduce rows through 256 MiB, with 64-256 MiB payload around `41.5-41.8 GB/s`;
     - log: `/home/steve/bench-results/qwen36-fp8-vllm/xccl-standalone-2rank-post-q4-localwrite-20260505T035142Z.log`.
   - revalidated the prior unsubmitted static FP8 TP4 n-gram candidate:
     - model: `vrfai/Qwen3.6-27B-FP8`;
     - TP4, FlashAttention2, compressed-tensors FP8, 512 prompt / 512 output, 3 measured iterations;
     - n-gram `num_speculative_tokens=4`, lookup min/max `2/4`;
     - with forced `CCL_ZE_IPC_EXCHANGE=sockets` and `CCL_TOPO_P2P_ACCESS=1`: `43.342333 tok/s`, so the earlier `50.193 tok/s` two-iteration screen did not reproduce under that CCL environment;
     - with closer-to-original CCL env (`CCL_ATL_TRANSPORT=ofi`, default IPC/topology): `47.674832 tok/s` output, `95.349664 tok/s` total;
     - latencies: `10.523862531`, `11.585105772`, `10.109288830` seconds;
     - JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T035653Z.json`;
     - LocalMaxxing accepted: `cmos3pnqo000kkz04o4aiup22`.
   - adjacent lookup max `5` under the same recovered/default CCL env was slower:
     - `42.159878 tok/s` output, `84.319756 tok/s` total;
     - JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T040134Z.json`.
   - decision: current validated FP8 best is lookup min/max `2/4` with default IPC/topology, not forced sockets and not lookup max `5`.
40. 2026-05-05 FP8 n-gram sweep around the validated TP4 best:
   - objective: test whether the `47.675 tok/s` static FP8 TP4 result was improved by changing speculative depth or lookup width while keeping the same CCL environment.
   - common configuration:
     - model: `vrfai/Qwen3.6-27B-FP8`;
     - vLLM/XPU `0.20.1`, FlashAttention2, compressed-tensors FP8;
     - selector `level_zero:0,1,2,3`, TP4/PP1;
     - `CCL_ATL_TRANSPORT=ofi`, default IPC/topology recognition;
     - 512 prompt / 512 output, 3 measured iterations after 1 warmup.
   - results:
     - n-gram `num_speculative_tokens=3`, lookup min/max `2/4`: `40.697016 tok/s`;
     - n-gram `num_speculative_tokens=4`, lookup min/max `2/3`: `43.130893 tok/s`;
     - n-gram `num_speculative_tokens=5`, lookup min/max `2/4`: `44.163969 tok/s`.
   - conclusion:
     - none beat the validated `47.674832 tok/s` n-gram `4`, lookup `2/4` run;
     - current FP8 next work should move to other levers, such as PP2 x TP2 drafter bugfix, real draft-model speculative decode, CCL/oneCCL env work, or vLLM XPU graph/communication constraints.
   - LocalMaxxing: not submitted because these are negative boundary runs.
41. 2026-05-05 vLLM PP2 x TP2 n-gram speculative unblock:
   - root cause:
     - `GPUModelRunner.__init__` only constructs `self.drafter` on the last pipeline-parallel rank;
     - `_build_attention_metadata()` dereferenced `self.drafter` on every PP rank whenever `speculative_config` was present;
     - PP0 ranks therefore crashed with `AttributeError: 'XPUModelRunner' object has no attribute 'drafter'`.
   - patch:
     - changed the metadata path to use `drafter = getattr(self, "drafter", None)`;
     - only Eagle/DFlash drafters need the `kv_cache_gid` branch; n-gram and non-drafter PP ranks can use the normal common metadata path;
     - applied to both `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py` and the active venv copy.
   - smoke result:
     - PP2 x TP2, 4x B70, static FP8, n-gram `4`, lookup `2/4`, 32 prompt / 8 output, 1 measured iteration;
     - completed successfully after the patch;
     - avg latency `0.32795706900651567 s`, output `24.393437 tok/s`;
     - JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in32-out8-bs1-20260505T043249Z.json`.
   - larger run status:
     - PP2 x TP2, 512 prompt / 256 output, n-gram `4`, lookup `2/4`;
     - failed during model load/memory accounting with `UR_RESULT_ERROR_DEVICE_LOST`, followed by oneCCL broken-pipe cleanup;
     - log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out256-bs1-20260505T043606Z.log`.
   - conclusion:
     - the vLLM drafter bug is fixed, but PP2 x TP2 remains a stability/runtime track rather than a speed track;
     - do not submit PP2 x TP2 results to LocalMaxxing until a full 512-token-class run completes.
42. 2026-05-05 vLLM PP2 x TP2 stability follow-up:
   - PP2 x TP2 non-speculative static FP8 now completes at 512 prompt / 128 output with lowered memory pressure:
     - `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=1024`, 2 measured iterations after 1 warmup;
     - avg latency `4.723338066491124 s`;
     - output `27.099479 tok/s`, total `135.497394 tok/s`;
     - JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260505T044235Z.json`.
   - PP2 x TP2 n-gram remains broken after the `self.drafter` patch:
     - n-gram `4`, lookup `2/4`, 512/128: scheduler emitted `num_scheduled_tokens=-3`, `total_num_scheduled_tokens=-3`;
     - n-gram `2`, lookup `2/4`, 512/128: scheduler emitted `num_scheduled_tokens=-1`, `total_num_scheduled_tokens=-1`.
   - rejected patch attempt:
     - changing the running-request schedule guard from `num_new_tokens == 0` to `num_new_tokens <= 0` prevents the negative-token assert but then leads to XPU `vectorized gather kernel index out of bounds`;
     - this suggests stale `-1` speculative placeholders still reach gather/sample code, so the true fix needs placeholder/spec-token cleanup, not just skipping negative scheduler work;
     - the guard was reverted.
   - LocalMaxxing: not submitted; PP2 non-spec is valid but not a useful speed result, and PP2+n-gram is a negative/runtime finding.
   - next action: leave PP2+n-gram quarantined until we can patch scheduler/output cleanup correctly; continue with validated TP4 FP8 and Q4 kernel/collective work.

43. 2026-05-05 Q4 allreduce-to-reshape experiment:
   - objective: remove the remaining slow plain allreduce cases that feed immediate `RESHAPE` nodes, especially the 48 `linear_attn_out-* -> RESHAPE` paths found in the previous partial probe.
   - patch:
     - added a backend proc hook `ggml_backend_comm_allreduce_to_tensor`;
     - added Meta backend detection for `PARTIAL` f32 tensors whose only consumer is a same-size mirrored `RESHAPE`;
     - added `GGML_META_FUSE_ALLREDUCE_RESHAPE=1` and optional `GGML_META_FUSE_ALLREDUCE_RESHAPE_LIMIT`;
     - added a SYCL single-kernel implementation that reduces partials and writes the mirrored reshaped output tensors directly.
   - one-token trace:
     - 4x B70, selector `level_zero:0,1,2,3`, 5120-element f32 allreduces;
     - `backend+reshape` selected for 48 reshape paths per token;
     - only final `attn_output-63` remained plain because its next consumer is `GET_ROWS`, not `RESHAPE`;
     - JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-reshape-fuse-probe-quad0123-p0n1-20260505T052621Z.jsonl`;
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-reshape-fuse-probe-quad0123-p0n1-20260505T052621Z.log`.
   - 4x B70 512/128 screen:
     - same-build fused-add control: `33.497463 tok/s`;
     - fused-add plus fused-reshape: `33.743952 tok/s`;
     - result is a marginal `+0.74%` screen, below the threshold for a meaningful quad-GPU improvement.
   - 3x B70 selector `2,1,3` validation:
     - fused-add plus fused-reshape, 512 prompt / 512 output, 3 reps: `43.734996 tok/s`;
     - prior fused-add-only validation remains better at `44.004344 tok/s`;
     - prompt throughput stayed effectively unchanged at `135.667935 tok/s`.
   - conclusion:
     - the graph recognition and backend handoff work;
     - this direct-to-reshape implementation is not a validated speed win and should remain experimental/env-gated;
     - do not submit to LocalMaxxing as an improved result.
44. 2026-05-05 MiniMax M2.7 large-model smoke blocker:
   - objective: after Qwen 27B showed useful 3x/4x B70 behavior, test the already-downloaded MiniMax M2.7 UD-IQ4_XS model as a larger four-GPU target.
   - command shape:
     - 4x B70, `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`;
     - `llama-bench -dev SYCL0,SYCL1,SYCL2,SYCL3 -sm row -ts 1/1/1/1 -ngl 99 -ncmoe 60 -fa 1 -ub 32 -ctk f16 -ctv f16 -p 0 -n 1 -r 1`;
     - MiniMax split/MoE debugging envs enabled, including `GGML_SYCL_MUL_MAT_ID_SPLIT=1` and host-bounce staging.
   - scheduler trace:
     - split 0 CPU `GET_ROWS` completes;
     - split 1 starts on `SYCL0` with 44 nodes, first `norm-0/RMS_NORM`;
     - input copies complete, then execution hangs or device-loses inside the first SYCL graph compute.
   - SYCL op trace:
     - `ggml_sycl_rms_norm` and following elementwise `MUL` complete;
     - the first failing operation is the dense attention projection `blk.0.attn_q.weight` `q8_0` `[3072,6144]` x `attn_norm-0` f32 `[3072,1]`;
     - default reordered MMVQ completes `quantize_row_q8_1_sycl` and then does not complete the q8_0 matvec;
     - forced DMMV completes `to_fp16_sycl` and then segfaults.
   - conclusion:
     - this is not yet a MiniMax `MUL_MAT_ID`/MoE split problem;
     - immediate blocker is the SYCL `q8_0 x vector` matvec path on the first dense attention projection;
     - next MiniMax work should build a targeted q8_0 matvec repro, add q8_0 kernel tracing, or add an env-gated fallback to reach the MoE path.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-05-minimax-q8-attn-matvec-blocker.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/minimax-m27-q8-attn-matvec-blocker-20260505.json`;
     - scheduler trace patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-scheduler-compute-trace-20260505.patch`.
   - LocalMaxxing: not submitted because no valid benchmark metric exists.
45. 2026-05-05 xe recovery blocker after device-lost:
   - trigger:
     - MiniMax q8_0 experiments produced Level Zero device-lost failures and a forced-DMMV segmentation fault;
     - subsequent Qwen multi-device runs failed below llama.cpp, including oneMKL GEMM `UR_RESULT_ERROR_DEVICE_LOST`.
   - recovery attempt:
     - PCI function reset was attempted for all four B70 VGA functions;
     - `sycl-ls` then aborted in NEO DRM initialization at `drm_neo.cpp:445`;
     - `xe` unbind/rebind deadlocked while binding `0000:83:00.0`.
   - current state:
     - only one B70 is visible to Level Zero;
     - `0000:83:00.0` is stuck in uninterruptible kernel sleep during bind;
     - `0000:a3:00.0` and `0000:e3:00.0` are unbound.
   - kernel stack:
     - `intel_edp_init_connector`;
     - `intel_dp_init_connector`;
     - `intel_setup_outputs`;
     - `xe_display_init_early`;
     - `xe_device_probe`.
   - recommendation:
     - reboot required;
     - before continuing benchmarks, load `xe` with `options xe disable_display=1 probe_display=0`;
     - this is appropriate for the ASPEED-console, headless-B70 compute setup and targets the observed display-probe deadlock.
   - post-reboot validation:
     - `sycl-ls` must show four Level Zero GPUs;
     - `/home/steve/sycl-peer-read-test` must pass with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`;
     - retry Qwen Q4 3-card `p16/n8` before full screens.
   - LocalMaxxing: not submitted because this is not a benchmark.
46. 2026-05-05 post-reboot GuC 70.49.4 Q4_0 validation:
   - system recovery:
     - `xe` is loaded headless with `disable_display=1` and `probe_display=0`;
     - all four B70s enumerate through Level Zero;
     - `/home/steve/sycl-peer-read-test` passes across all four GPUs;
     - kernel logs confirm BMG GuC `70.49.4` on all four B70s.
   - fresh DNN build:
     - built current source with `GGML_SYCL_DNN=ON` and `GGML_SYCL_DEVICE_ARCH=intel_gpu_bmg_g31` at `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-aot-dnn-current`;
     - 2x and 3x tiny prompt/decode smokes survive without reset;
     - DNN is a stability/debug fallback, not the current speed path.
   - validated fast path:
     - Qwen3.6 27B Q4_0 GGUF, 3x B70 selector `2,1,3`;
     - exact fast env includes Q8 cache, async copy, single-kernel allreduce, event barrier, and `GGML_META_FUSE_ALLREDUCE_ADD=1`;
     - 512 prompt / 512 output, 3 repeats: prompt `135.705541 tok/s`, decode `44.180797 tok/s`, total `66.659637 tok/s`;
     - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-guc70494-nondnn-build-triple213-exactfast-p512n512-r3-20260505T121540Z.jsonl`;
     - LocalMaxxing: `cmoslhw0i0008jj04h59bb96n`.
   - decision:
     - current Q4_0 reference is still reproducible post-reboot and post-GuC update;
     - continue optimizing from this non-DNN SYCL path for Q4_0 speed work;
     - use DNN only to isolate prompt-side oneMKL GEMM stability issues.
47. 2026-05-06 FP8 TP2/2x2 layout check:
   - target:
     - determine whether Qwen3.6 27B FP8 can run as two independent 2-card tensor-parallel groups on the four B70s.
   - static compressed-tensors FP8:
     - `/home/steve/models/qwen3.6-27b-fp8-vrfai`, `TP=2`, `PP=1`, selector `level_zero:0,1`;
     - fails to load from XPU OOM during model parameter allocation, including with lower `GPU_MEM_UTIL` and `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`.
   - dynamic FP8 HF shard set:
     - `/home/steve/models/qwen3.6-27b-fp8-hf`, `TP=2`, `PP=1`;
     - current exact block-FP8 fallback fails while materializing BF16 weights in `XPUBF16Fp8BlockScaledMMLinearKernel.process_weights_after_loading`.
   - opt-in local requant path:
     - `VLLM_XPU_BLOCK_FP8_REQUANT=1` selects `XPURequantFp8BlockScaledMMLinearKernel`;
     - this loads TP2/PP1, but it requantizes 128x128 block-FP8 weights into XPU W8A16 FP8 format and is therefore a quality-risk experiment;
     - 512/128 no-spec smoke: `9.098410 tok/s`;
     - 512/512 n-gram 4 lookup `2/4`: `24.498929 tok/s`;
     - decision: do not treat as a speed path or submit to LocalMaxxing.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-fp8-tp2-requant-negative.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/qwen36-fp8-tp2-requant-negative-20260506.json`;
     - env registration patch: `/home/steve/llm-optimization-artifacts/patches/vllm-xpu-block-fp8-requant-env-20260506.patch`.
   - next:
     - keep 2x2 serving on hold until native block-scaled FP8 GEMM exists or requant passes quality checks and becomes much faster;
     - continue with TP4 FP8 and Q4_0 GGUF paths for single-session speed.
48. 2026-05-06 Q4_0 reordered MMVQ VDR/reduction screen:
   - target:
     - test two quality-preserving kernel scheduling knobs in the hot Q4_0 reordered MMVQ decode path;
     - keep the Q4_0 model and dequant math unchanged.
   - source controls added:
     - `GGML_SYCL_REORDER_MMVQ_Q4_VDR=1|4` to test lane/block work grouping against the default VDR=2;
     - `GGML_SYCL_REORDER_MMVQ_XOR_REDUCE=1` to test explicit XOR subgroup reduction against `sycl::reduce_over_group`;
     - default runtime remains unchanged.
   - single-card screen:
     - Qwen3.6 27B Q4_0 GGUF, `ONEAPI_DEVICE_SELECTOR=level_zero:2`;
     - `-p 0 -n 128 -r 2`, `-fa 1`, `-ub 32`, f16 KV cache, Q8 cache, fused MMVQ2;
     - default VDR=2: `24.654646 tok/s`;
     - VDR=1: `24.516621 tok/s`;
     - VDR=4: `23.232102 tok/s`;
     - default reduction: `24.599359 tok/s`;
     - XOR reduction: `24.557468 tok/s`.
   - conclusion:
     - this is not a useful optimization path;
     - VDR=2 and `sycl::reduce_over_group` remain the best tested choices;
     - next Q4 kernel work should focus on a deeper ESIMD/XMX Q4_0 x Q8_1 matvec path or on graph/communication fusion rather than this lane grouping.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-vdr-xor-negative.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/q4-vdr-xor-negative-20260506.json`;
     - patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-vdr-xor-negative-20260506.patch`.
   - LocalMaxxing: not submitted because this is a negative microbenchmark, not an improved model run.
49. 2026-05-06 Q4_0 pair/triple/quad topology screen:
   - target:
     - explain whether the four-card Q4_0 regression is caused by a bad GPU, a bad pair, root ordering, or the 4-way tensor-parallel/allreduce path itself.
   - setup:
     - Qwen3.6 27B Q4_0 GGUF;
     - llama.cpp SYCL AOT BMG G31 build, tensor split mode;
     - `-p 512 -n 128 -r 2`, `-fa 1`, `-ub 32`, f16 KV cache, Q8 cache, fused MMVQ2, fused allreduce+add, single-kernel allreduce, event barrier, and `GGML_SYCL_COMM_SYNC_AFTER=2`.
   - ordered two-card sweep:
     - all 12 ordered pairs landed tightly between `40.687815` and `40.747873 tok/s`;
     - no pair or root ordering stood out as bad;
     - summary: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pair-order-sweep-devslash-p512n128-r2-20260506T113813Z.jsonl`.
   - triple/quad sweep:
     - `2,1,3`: `45.195064 tok/s`;
     - `0,1,2`: `45.415802 tok/s`;
     - `0,1,3`: `45.140951 tok/s`;
     - `0,2,3`: `46.037050 tok/s`;
     - `2,1,3,0`: `34.550830 tok/s`;
     - `0,1,2,3`: `34.722804 tok/s`;
     - summary: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-triple-quad-order-sweep-p512n128-r2-20260506T115036Z.jsonl`.
   - conclusion:
     - the 4-card regression is not a bad B70, a bad pair, or a simple root-ordering problem;
     - every ordered pair is healthy and every tested triple is healthy, including triples with device 0;
     - next 4x Q4_0 work should target the 4-way allreduce/scheduling implementation or reduce the number of per-token collectives.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-topology-4way-bottleneck.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/q4-topology-4way-bottleneck-20260506.json`.
   - LocalMaxxing: not submitted because this is a diagnostic topology screen, and the already submitted 3-card and 4-card full 512/512 runs are better leaderboard-quality records.
50. 2026-05-06 Q4_0 graph-consumer and vec4 allreduce follow-up:
   - combined allreduce consumer fusion:
     - tested `GGML_META_FUSE_ALLREDUCE_RESHAPE=1` and `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` with the existing fused allreduce+add path;
     - 4x short `p512/n128/r2`: off `33.987384 tok/s`, on `34.874597 tok/s`;
     - 4x full `p512/n512/r3` with the extra fusions: `34.842877 tok/s`, which does not beat the existing 4x best `34.929313 tok/s`;
     - decision: keep available for diagnostics, but not a best path.
   - temporary vec4 allreduce experiment:
     - tested `GGML_SYCL_COMM_VEC4_F32=1`, vectorizing the small F32 reduction loops with `sycl::vec<float, 4>`;
     - 3x short control `45.482621 tok/s`, vec4 `45.313034 tok/s`;
     - 4x short control `34.890219 tok/s`, vec4 `34.198055 tok/s`;
     - result is negative, especially on 4x.
   - cleanup:
     - removed the temporary `GGML_SYCL_COMM_VEC4_F32` source gate and vector branches;
     - rebuilt `llama-bench` and `llama-cli`;
     - post-revert 3x smoke on selector `2,1,3` returned `45.668525 tok/s` for `p512/n128/r1`.
   - conclusion:
     - simple scalar/vector formatting of the tiny F32 allreduce is not the 4x bottleneck;
     - the useful Q4 path is still deeper graph/communication work: true output-projection plus allreduce epilogue, or reducing the count of per-token collectives.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-combofuse-vec4-negative.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/q4-combofuse-vec4-negative-20260506.json`.
   - LocalMaxxing: not submitted because this is neutral/negative diagnostic data, not a new best model benchmark.
51. 2026-05-06 FP8 PP2 GDN speculative fallback:
   - objective:
     - continue the 2x2 FP8 layout investigation by fixing the concrete PP2 n-gram failure instead of treating PP as generally broken.
   - deterministic benchmark controls added:
     - `VLLM_BENCH_LATENCY_PROMPT_SEED=<int>`;
     - `VLLM_BENCH_LATENCY_PROMPT_MODE=repeat`;
     - `VLLM_BENCH_LATENCY_REPEAT_PERIOD=<int>`.
   - wrapper/source hygiene:
     - `/home/steve/bench-vllm-qwen36-fp8.sh` now imports `/home/steve/src/vllm` through `PYTHONPATH` when present;
     - this avoids mixing copied source files into the installed venv package.
   - source finding:
     - PP handoff is working: last PP rank can propose draft tokens and PP0 receives scheduled speculative work;
     - the failing forced-repeat trace scheduled `spec_lens={'0-a172e5cc': 4}` on PP0;
     - the immediate crash was the explicit XPU GDN assertion in `_gdn_attention_core_xpu_impl`: `attn_metadata.spec_sequence_masks is None`;
     - this is a GDN XPU custom-op speculative support gap, not the earlier PP ownership bug.
   - patch:
     - added a generic GDN fallback in `_gdn_attention_core_xpu_impl()` for speculative metadata;
     - when `spec_sequence_masks` is present, it reconstructs `mixed_qkv`, `z`, `b`, and `a` from the projected XPU tensors and calls `self._forward_core(...)`;
     - non-speculative GDN still uses the native `_xpu_C.gdn_attention` path.
   - validation status:
     - `--enforce-eager` completed but did not schedule draft tokens in that sample, so it was not a real validation;
     - the patched non-eager retry failed during model load with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, followed by `UR_RESULT_ERROR_DEVICE_LOST`;
     - post-reboot forced-repeat retry completed with real PP speculation: PP1 proposed `draft_lens=[4]`, PP0 scheduled `spec_lens` with 4 draft tokens, and the old GDN `spec_sequence_masks` assertion did not fire;
     - the successful validation was short and slow: 32 output tokens at `8.723851 tok/s`, so it is not a speed path;
     - the next longer PP2 load failed before inference with `UR_RESULT_ERROR_DEVICE_LOST` during embedding weight copy, confirming repeated vLLM FP8 load/unload remains unstable;
     - post-failure sanity checks showed all four B70s still enumerate and small Torch XPU allocations work on every device;
     - a post-failure Q4_0 llama.cpp 3-card sanity run also completed at `45.104294 tok/s` for `p512/n128/r1`;
     - do not submit these runs to LocalMaxxing.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-vllm-pp2-gdn-spec-fallback.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/qwen36-fp8-pp2-gdn-spec-fallback-20260506.json`;
     - patch: `/home/steve/llm-optimization-artifacts/patches/vllm-xpu-gdn-spec-fallback-and-deterministic-bench-20260506.patch`.
   - next:
     - keep the GDN fallback patch, but stop treating PP2 as a near-term speed path until the repeated-load `DEVICE_LOST` issue is solved;
     - if returning to PP2, use one long-lived server/process or add a safer unload/reset path instead of repeated `bench latency` process loads;
     - continue Q4_0 GGUF and TP4 FP8 as the active speed paths while PP2 GDN speculation remains experimental.
52. 2026-05-06 Q4_0 four-card assist split:
   - target:
     - recover useful four-card Q4_0 performance after reboot without changing model quality or GPU power limits.
   - finding:
     - equal four-card tensor split remains inefficient;
     - a fourth-card assist split is materially better, confirming the fourth B70 is healthy but equal four-way row shards are too narrow for the current reordered Q4_0 MMVQ and communication path.
   - validated run:
     - selector `level_zero:2,1,3,0`;
     - devices `SYCL0/SYCL1/SYCL2/SYCL3`;
     - tensor split `1/1/1/0.05`;
     - `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2`;
     - `-p 512 -n 512 -r 3`, Q4_0 weights, f16 KV cache, flash attention enabled, speculative decoding disabled;
     - prompt `80.906412 tok/s`;
     - decode `39.204149 tok/s`;
     - total `52.815789 tok/s`;
     - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-sg2-assist005-quad2130-p512n512-r3-20260506T141453Z.jsonl`.
   - comparison:
     - previous full four-card equal split: `34.929313 tok/s`;
     - assist split gain: `+4.274836 tok/s`, `+12.24%`;
     - best current three-card Q4_0 result remains faster at `46.194319 tok/s`.
   - LocalMaxxing:
     - accepted ID `cmou581wv002dld0197mffpco`;
     - first payload with backend `sycl-level-zero` returned HTTP 400, accepted retry omitted the backend enum and preserved SYCL/Level Zero details in notes.
   - artifacts:
     - note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-4x-assist-split.md`;
     - data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-4x-assist-split-20260506.json`;
     - plan addendum: `/home/steve/llm-optimization-artifacts/plans/2026-05-06-q4-4x-assist-split-addendum.md`.
   - next:
     - do not treat four-card Q4_0 as the production path until it beats three-card;
     - focus Q4_0 work on narrow-shard reordered MMVQ efficiency and output-projection/allreduce/residual epilogue fusion;
     - keep FP8/vLLM TP4 as the current stronger all-four-card speed track.
53. 2026-05-06 Q4_0 four-card follow-up probes:
   - fine assist-ratio sweep:
     - TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/tensorsplit-quad-sg2-fine-assist-ratio-p0n128-r2-20260506T143835Z.tsv`;
     - `1/1/1/0.01` and `1/1/1/0.02` failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during `MUL_MAT`;
     - surviving ratios from `0.03` through `0.12` landed in the `37.762326` to `38.822236 tok/s` range;
     - conclusion: further ratio tuning does not recover the missing four-card scaling.
   - Q4 fused2 status:
     - debug trace `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-fused2-triple213-p0n1-r1-20260506T144944Z.log`;
     - counted `480` fused2 calls and `1,584` plain reordered MMVQ calls under debug;
     - conclusion: fused2 is already active under tensor split, so do not spend time on a pure enablement patch.
   - allreduce stats:
     - stats log `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-stats4-assist005-quad2130-p0n1-r1-20260506T145113Z.log`;
     - four-card assist split does `128` reductions per generated token, each `20,480` bytes;
     - warm allreduce time is `4.213 ms/token`, `32.913 us` average;
     - first/cold allreduce time is `6.724 ms/token`, `52.528 us` average;
     - most reductions use the `backend+add` fused residual path, but the collective count is unchanged at two per layer.
   - next:
     - run a focused communication-flag sweep around `skip_root_ready`, root rotation, local-write, pairwise, striped, and small-f32 paths;
     - if none beats the current single-kernel path, the next source patch should try a lower-latency 20 KB allreduce-add specialization for four B70s or graph-level collective-count reduction.
54. 2026-05-06 Q4_0 four-card communication flag sweep:
   - TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/comm-flag-sweep-quad-assist005-p0n128-r2-20260506T145544Z.tsv`;
   - baseline `sync_after=2`: `38.195956 tok/s`;
   - best short variant was `sync_after=0`: `38.354083 tok/s`, which is too small to treat as a validated improvement;
   - `skiproot_fuseaddroot`: `38.218925 tok/s`;
   - `skip_root_ready`: `37.971362 tok/s`;
   - `fuseadd_root_residual`: `37.898128 tok/s`;
   - `local_write`: `37.208619 tok/s`;
   - `rotate_root`: `37.087460 tok/s`;
   - clearly bad paths:
     - `pairwise4`: `28.156880 tok/s`;
     - `striped4`: `26.468293 tok/s`;
     - `no_fuseadd_smallf32`: `28.999878 tok/s`.
   - conclusion:
     - existing communication toggles do not explain the missing four-card scaling;
     - keep validated command on the current single-kernel allreduce-add path;
     - next Q4_0 source work should inspect MMVQ row-shard efficiency and collect per-stage timing for 3x vs 4x before patching.
55. 2026-05-06 Q4_0 narrow-shard follow-ups:
   - MMV_Y=2 build:
     - 3x default `MMV_Y=1`: `44.321101 tok/s`;
     - 3x `MMV_Y=2`: `44.366675 tok/s`;
     - 4x assist `MMV_Y=2`: `38.181852 tok/s`;
     - decision: do not promote `MMV_Y=2`.
   - MUL_MAT stage instrumentation:
     - 3x: `1206` quantize calls and `2982` matmul kernel calls;
     - 4x assist: `1488` quantize calls and `3520` matmul kernel calls;
     - total matmul byte volume is essentially unchanged at about `30.18 GB`;
     - conclusion: the fourth assist shard adds launch and quantization overhead without enough useful Q4 work.
   - Explicit trailing-zero split:
     - `-ts 1/1/1/0` aborts at `ggml-backend.cpp:120: GGML_ASSERT(buffer) failed`;
     - treat zero-width trailing split as a llama.cpp/SYCL bug, not a valid optimization.
   - Experimental skip-last patch:
     - patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-split-skip-last-below-rows-focused-20260506.patch`;
     - env: `GGML_SYCL_SPLIT_SKIP_LAST_BELOW_ROWS`;
     - patched default 3x sanity: `44.778557 tok/s`;
     - best threshold screen was `6144` rows at `38.153344 tok/s`, below the validated `39.204149 tok/s` 4x assist run;
     - decision: safe diagnostic only; keep env unset for production.
   - next:
     - stop simple split-ratio and communication-flag sweeps for 4x Q4_0 unless a profiler identifies a narrower cause;
     - focus on reducing the number of per-token collectives or fusing output projection/allreduce/residual work;
     - consider using the fourth B70 for speculative draft work or a second session while 3x remains the main Q4_0 speed path.
56. 2026-05-06 Q4_0 speculative draft follow-ups:
   - downloaded a small draft candidate:
     - repo: `llmware/qwen-3.5-4b-gguf`;
     - file: `/home/steve/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`;
     - size: `2.6G`.
   - target 3x plus draft on fourth B70:
     - selector `level_zero:2,1,3,0`;
     - target `-dev SYCL0,SYCL1,SYCL2 -sm tensor -ts 1,1,1`;
     - draft `--spec-draft-device SYCL3 --spec-draft-ngl 99`;
     - isolated `--fit off`, `n=1` run still failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`;
     - interpretation: exposing the fourth B70 for the draft still makes target tensor split interact with global SYCL device count, so target 3x plus draft 1x needs source work around active-device-aware split buffers.
   - CPU draft control:
     - completed but was invalid as a speed path;
     - `n_drafted=128`, `n_accept=0`, `accept=0.000%`;
     - decode section: `33` tokens in `3.747 s`, `8.807 tok/s`;
     - log had repeated inconsistent sequence-position failures.
   - target-only n-gram:
     - `llama-speculative` requires `--model-draft`;
     - `llama-cli --spec-type ngram-mod` timed out after `180 s` before useful timings.
   - decision:
     - do not use generic Qwen3.5 4B as a Qwen3.6 27B draft for speed claims;
     - prefer Qwen3.6-specific MTP/DFlash draft heads or vLLM FP8 speculative paths;
     - if returning to llama.cpp target+draft, first fix SYCL split-buffer active-device accounting so the target can use 3 B70s while the draft owns the fourth.
57. 2026-05-06 Q4_0 speculative draft placement patch:
   - patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-speculative-draft-single-device-splitmode-20260506.patch`;
   - patched `examples/speculative/speculative.cpp` so explicit `--spec-draft-device none` or a single draft device no longer inherits the target's `LLAMA_SPLIT_MODE_TENSOR`;
   - placement fix validated:
     - target stays on `Meta()` over `SYCL0,SYCL1,SYCL2`;
     - draft `--spec-draft-device SYCL3` loads model/KV/compute on `SYCL3`;
     - draft `--spec-draft-device none` loads CPU model/KV/compute instead of creating a four-device meta backend.
   - remaining failure:
     - CPU draft and `SYCL3` draft both timed out;
     - repeated failure signal: `alloc: can't allocate 889986048 Bytes of memory on device/GPU`;
     - disabling `GGML_SYCL_Q8_CACHE` and reducing context/batch to `-c 512 -b 512 -ub 16` did not change the allocation size;
     - Qwen3.5 4B does run in `llama-bench` on `SYCL3`, but `llama-cli`/`llama-speculative` common-init paths hang or time out.
   - decision:
     - keep the patch as a reproducibility artifact;
     - do not submit these failed speculative runs to LocalMaxxing;
     - stop treating generic Qwen3.5 4B as a useful Qwen3.6 27B draft candidate;
     - prefer Qwen3.6-specific MTP/DFlash or vLLM FP8 speculative paths.
58. 2026-05-06 FP8 MTP hybrid follow-up:
   - dynamic FP8 directory `/home/steve/models/qwen3.6-27b-fp8-hf` contains real `mtp.safetensors`;
   - static compressed-tensors directory `/home/steve/models/qwen3.6-27b-fp8-vrfai` has no `mtp.*` tensors, so earlier static MTP runs were not clean MTP data;
   - created hybrid symlink directory `/home/steve/models/qwen3.6-27b-fp8-vrfai-mtp-hybrid` with static main weights plus dynamic `mtp.safetensors`;
   - vLLM confirmed checkpoint size `33.90 GiB` and loaded `2/2` safetensors shards;
   - patched `vllm/model_executor/models/qwen3_5_mtp.py` so packed-name matching uses the original checkpoint tensor name for each candidate:
     - patch: `/home/steve/llm-optimization-artifacts/patches/vllm-qwen35-mtp-loader-original-name-20260506.patch`;
     - this removed bogus doubled names like `qkqkv_proj` and `gate_gate_up_proj`.
   - measured 32 prompt / 8 output TP4 smoke results:
     - dynamic FP8 real MTP eager: `0.960369 tok/s`;
     - hybrid MTP eager before loader patch: `7.409132 tok/s`;
     - hybrid MTP eager after loader patch: `7.818582 tok/s`;
     - hybrid MTP compiled/async after loader patch: `0.581715 tok/s`.
   - remaining issue:
     - packed scale tensors are still skipped for `qkv_proj`, `gate_up_proj`, `down_proj`, and `o_proj`;
     - this appears to be missing packed scale parameters in the compressed-tensors MTP parameter dict, not the previous string-mutation bug.
   - decision:
     - do not submit these MTP smokes to LocalMaxxing;
     - keep MTP as a source/debug track, but do not run longer MTP speed claims until packed scale loading is clean;
     - validated non-spec static FP8 TP4 and Q4_0 TP3 remain the useful performance baselines.
59. 2026-05-06 FP8 MTP block-FP8 clean-load follow-up and llm-scaler source mining:
   - fixed the vLLM TP wrapper so TP4 workers see all four B70s:
     - previous wrapper defaulted `ONEAPI_DEVICE_SELECTOR=level_zero:0`;
     - vLLM/XPU workers then saw `device_count=1`, causing TP4 placement/index failures;
     - wrapper now defaults selector to empty and unsets `ONEAPI_DEVICE_SELECTOR` unless the caller explicitly sets one.
   - patched Qwen3.5 MTP handling for the hybrid model:
     - env: `VLLM_QWEN35_MTP_FORCE_FP8_BLOCK=1`;
     - target model stays static compressed-tensors FP8;
     - MTP drafter layers get a local block-FP8 `Fp8Config` with dynamic activations and `weight_block_size=[128,128]`;
     - `mtp.fc` is ignored and remains BF16;
     - `VLLM_QWEN35_MTP_FORCE_FP8_BLOCK` is registered in `vllm/envs.py`.
   - corrected MTP load status:
     - logs now select `XPURequantFp8BlockScaledMMLinearKernel for Fp8LinearMethod`;
     - no missing `weight_scale_inv` warnings;
     - no bogus `qkqkv_proj` or `gate_gate_up_proj` names.
   - corrected MTP performance remains poor:
     - eager smoke, 32 prompt / 8 output: `2.364238 tok/s`;
     - compiled/async smoke, 32 prompt / 32 output: `1.844712 tok/s`;
     - both are far below non-spec static FP8 TP4 baselines, so do not submit to LocalMaxxing.
   - llm-scaler review:
     - local clone `/home/steve/src/llm-scaler` is at `e0b0703`, tag `vllm-0.14.0-b8.2.1`;
     - high-value reference files are `custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/int4_GEMV.h` and `resadd_norm_gemv_int4.h`;
     - useful ideas: ESIMD decode GEMV for BMG, `K_SPLIT` for small-N/high-K decode, fused multi-GEMV dispatch to save launch overhead, and fused residual-add + RMSNorm + INT4 GEMV;
     - caveat: its "GGML q4_0" path uses a group-size-128 layout with scale shape `[N, K/128]`, while llama.cpp GGUF `Q4_0` uses block size 32, so it is not a drop-in kernel.
   - next source direction:
     - mine llm-scaler for BMG ESIMD scheduling and fusion patterns, not direct Q4_0 format compatibility;
     - continue Q4_0 work around deeper graph/kernel fusion and per-token launch/quantization reduction, especially RMSNorm/GEMV or multi-GEMV fusion, because communication flag sweeps are exhausted.
60. 2026-05-06 Q4_0 graph-pattern probe:
   - added diagnostic env `GGML_META_GRAPH_PATTERN_STATS`;
     - unset: no behavior change;
     - `1`: summary only, once per graph UID;
     - `2`: summary plus repeated-MUL_MAT shared-activation examples;
     - `3`: repeat every compute call, useful only for debugging rebuild churn.
   - build:
     - rebuilt `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench`;
     - patch artifact: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-meta-graph-pattern-stats-current-20260506.patch`.
   - validated run:
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/meta-graph-pattern2-triple213-p512n1-r1-20260506T175517Z.log`;
     - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/meta-graph-pattern2-triple213-p512n1-r1-20260506T175517Z.jsonl`;
     - command used the known 3x Q4 path, `-p 512 -n 1`, flash attention, f16 KV, tensor split `1/1/1`, selector `level_zero:2,1,3`.
   - probe result per graph:
     - `3656` nodes;
     - `497` `MUL_MAT` nodes;
     - `344` Q4_0 `MUL_MAT` nodes;
     - `127-128` partial `MUL_MAT` nodes;
     - `209` `RMS_NORM -> MUL` chains;
     - `129` norm-fed matmul groups;
     - `369` norm-fed matmul edges;
     - `128` `ADD -> RMS_NORM -> MUL -> MUL_MAT*` groups;
     - `128` repeated-activation groups covering `368` matmuls.
   - examples:
     - `attn_post_norm-N` feeds `ffn_gate.weight:q4_0` and `ffn_up.weight:q4_0` in every layer;
     - many `attn_norm-N` activations feed multiple attention projections, sometimes including `attn_qkv.weight:q4_0`, `attn_gate.weight:q4_0`, and f32 SSM alpha/beta projections;
     - Qwen3.6 layers alternate between combined QKV and separate Q/K/V shapes, so a fused path must tolerate both grouped and split attention projection layouts.
   - conclusion:
     - this confirms the next quality-preserving Q4 target is multi-GEMV / norm+GEMV fusion for same-activation matmuls;
     - FFN gate/up fusion is the cleanest first integration target because it is a pair of Q4_0 projections sharing `attn_post_norm`;
     - attention projection fusion is also promising but more shape/layout dependent;
     - do not submit this to LocalMaxxing because it is diagnostic, not a throughput result.
61. 2026-05-06 post-reboot Q4 sanity and FP8 32k-context topology follow-up:
   - Q4_0 post-reboot sanity:
     - three-B70 path, selector `level_zero:2,1,3`, tensor split `1/1/1`;
     - `512` prompt / `512` output / `3` reps;
     - result: `45.638154 tok/s`;
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/postreboot-triple213-q4-syncafter2-fusemmvq2-p512n512-r3-20260506T180324Z.log`;
     - this is close to the submitted `46.194319 tok/s` best, so it was not resubmitted.
   - FP8 TP2/PP2 32k-context validation:
     - model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
     - TP2/PP2, `max_model_len=32768`, `2048` prompt / `256` output;
     - result: `26.361533 tok/s`;
     - reported KV cache: `1,056,196` tokens, max `32.23x` concurrency at 32,768 context;
     - stable but much slower than TP4 for batch-1 decode.
   - FP8 TP2/PP2 with `VLLM_XPU_ENABLE_XPU_GRAPH=1`:
     - result: `25.582015 tok/s`;
     - log says XPU graph disables cudagraph mode because communication op capture is unsupported;
     - decision: do not spend more runs on XPU graph for TP/PP communication paths.
   - FP8 TP4 32k-context validation:
     - TP4/PP1, `max_model_len=32768`, `2048` prompt / `256` output;
     - result: `42.996276 tok/s`, `386.966486 total tok/s`;
     - reported KV cache: `1,133,163` tokens, max `34.58x` concurrency at 32,768 context;
     - this is faster and has more reported 32k KV capacity than TP2/PP2 for Qwen3.6 27B.
   - FP8 TP4 32k-context with CPU n-gram:
     - spec config: n-gram, `num_speculative_tokens=4`, lookup min/max `2/4`;
     - result: `40.123211 tok/s`;
     - slower than no-spec because n-gram disables async scheduling and had poor value on the random 2048-token bench prompt.
   - LocalMaxxing:
     - submitted the clean TP4 32k no-spec result;
     - ID: `cmoudx2qr00c3ld01xxq8hiu0`;
     - status: `APPROVED`;
     - API note: `backend=xpu` is rejected by current LocalMaxxing validation, so XPU/Level Zero was recorded in `engineFlags.extraFlags`.
   - PCIe reporting observation:
     - sudo `lspci -vvv` reports every B70 endpoint with `LnkCap Speed 2.5GT/s, Width x1` and `LnkCap2` only `2.5GT/s`;
     - this is suspicious for B70 and may be immature device/firmware reporting, but it agrees with oneCCL warnings that topology is PCIe rather than fabric;
     - keep treating PCIe/allreduce overhead as a likely reason 4x Q4_0 equal split underperforms.
   - decision:
     - for Qwen3.6 27B FP8, prefer TP4/PP1 over TP2/PP2 for single-session speed and 32k context;
     - keep PP2 as a fallback for larger models or future pipeline-parallel experiments, not as the primary 27B path;
     - keep Q4_0 optimization focused on deeper llama.cpp same-activation multi-GEMV / norm+GEMV fusion.
62. 2026-05-06 active-device row-split patch and row-split safety screen:
   - implemented a focused llama.cpp patch in `src/llama-model.cpp` so `LLAMA_SPLIT_MODE_ROW` split-buffer creation maps selected model devices back to physical backend-registry indices before calling `ggml_backend_split_buffer_type`;
   - motivation:
     - llama.cpp `tensor_split` is selected-device-order;
     - SYCL split buffers interpret `tensor_split` as global physical SYCL device order;
     - with four visible B70s, a target using only two or three B70s could unintentionally assign row shards to an unselected B70.
   - build:
     - `llama-bench` and `llama-cli` rebuilt successfully in `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`;
     - patch artifact: `/home/steve/llm-optimizations/patches/llama-cpp-active-device-row-split-current-20260506.patch`.
   - tensor-split sanity after patch:
     - Qwen3.6 27B Q4_0 GGUF, three B70s `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`;
     - `128` prompt / `128` output;
     - decode result: `45.065268 tok/s`;
     - prefill result: `106.106708 tok/s`;
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sanity-tensor3-after-active-split-20260506T183355Z.log`.
   - row-split safety screen:
     - Qwen3.5 4B Q4_K_M, row split on `SYCL2/SYCL3`, `32` prompt / `8` output;
     - still failed with Level Zero `UR_RESULT_ERROR_DEVICE_LOST` inside `oneapi::mkl::blas::column_major::gemm` from `ggml_sycl_op_mul_mat_sycl`;
     - kernel log recorded an `xe 0000:a3:00.0` GT reset at `2026-05-06 14:32:41 America/Toronto`;
     - log: `/home/steve/bench-results/active-split/qwen35-4b-row-sycl23-smoke-20260506T183238Z.log`.
   - decision:
     - active-device accounting is patched and should be kept for reproducibility;
     - row split remains unsafe for throughput work until the SYCL split matmul path is fixed;
     - do not use row split as the production Q4 path and do not submit this diagnostic to LocalMaxxing.
63. 2026-05-06 Q4_0 fused MMVQ2 + split SwiGLU patch:
   - implemented an opt-in SYCL path behind `GGML_SYCL_FUSE_MMVQ2_SWIGLU=1`;
   - target graph:
     - two Q4_0 FFN gate/up matvecs;
     - same F32 activation input;
     - split `GGML_OP_GLU` with `GGML_GLU_OP_SWIGLU`;
     - contiguous non-split SYCL buffers;
     - direct write of `silu(gate) * up` to the GLU output tensor;
   - build:
     - rebuilt `llama-bench` and `llama-cli` successfully in `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`;
     - patch artifact: `/home/steve/llm-optimizations/patches/llama-cpp-sycl-fused-mmvq2-swiglu-current-20260506.patch.gz.b64`;
   - correctness:
     - `llama-completion`, greedy decode, same prompt/seed, 8 generated tokens;
     - baseline and fused stdout SHA256 both `a7514e8196ec963459785822b3fcf25b1743096a4bdd5ec746225a7c9a29be19`;
     - this is quality-preserving for the tested decode path: same target model, same Q4_0 weights, same f16 KV, no speculative decoding, no power changes;
   - single-B70 512/512:
     - baseline fused MMVQ2: `24.567164 tok/s`;
     - fused MMVQ2 + SwiGLU: `24.657839 tok/s`;
     - gain: `+0.37%`;
     - conclusion: correct but not enough to close the single-B70 Linux gap to the Windows Q4_0 result;
   - three-B70 tensor split 512/512:
     - devices: `SYCL2/SYCL1/SYCL3`;
     - split: `-sm tensor -ts 1/1/1`;
     - required `-ub 128` because default `-ub 512` now fails graph reservation with `GGML_ASSERT(buffer) failed` in `ggml_backend_buffer_get_size()` while allocating meta compute buffers;
     - baseline `-ub 128`: `45.745560 tok/s`;
     - fused MMVQ2 + SwiGLU `-r 2 -ub 128`: `46.804859 tok/s`, `75.217668 total tok/s`;
     - LocalMaxxing ID: `cmougm58m00dpld012rbm9rbs`;
   - decision:
     - keep the fused SwiGLU env gate as a valid incremental win;
     - treat `-ub 128` as the current required 3-B70 512/512 validation knob;
     - next source-level target should fuse a broader same-activation GEMV group or norm+projection path because FFN SwiGLU launch removal alone is too small on one B70.
64. 2026-05-06 Q4_0 RMS_NORM + scale-MUL fusion:
   - implemented an opt-in SYCL path behind `GGML_SYCL_FUSE_RMS_NORM_MUL=1`;
   - target graph:
     - F32 `RMS_NORM`;
     - immediately consumed by F32 `MUL`;
     - other MUL input is a contiguous one-dimensional F32 scale tensor;
     - all tensors on the same non-split SYCL device buffer;
     - direct write of normalized-and-scaled output to the MUL destination;
   - added allocator diagnostics while investigating the 3x `-ub 512` failure:
     - meta buffer allocation now logs the failing simple backend and size before returning allocation failure;
     - `ggml_vbuffer_alloc` logs the failing chunk/buft/size;
     - confirmed `-ub 512` fails while reserving a roughly `529530880` byte meta compute buffer, with `SYCL3` failing the allocation;
   - build:
     - rebuilt `llama-bench` and `llama-completion` successfully in `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`;
     - patch artifact: `/home/steve/llm-optimizations/patches/llama-cpp-sycl-rmsnormmul-current-20260506.patch.gz.b64`;
   - correctness:
     - `llama-completion`, greedy decode, same prompt/seed, fused MMVQ2 and fused SwiGLU enabled on both sides;
     - baseline and fused stdout SHA256 both `f7254271342a273042f88b21af7267f2fe5a06340ba68a9fc765746090a645aa`;
     - debug 1-token run confirmed `418` RMS_NORM+MUL fused calls;
     - this is quality-preserving for the tested decode path: same target model, same Q4_0 weights, same f16 KV, no speculative decoding, no sampling change, no power changes;
   - single-B70 512/512:
     - devices: `SYCL2`;
     - decode: `24.960284 tok/s`;
     - previous fused MMVQ2+SwiGLU single result: `24.657839 tok/s`;
     - total throughput: `47.655433 tok/s`;
   - two-B70 tensor split 512/512:
     - devices: `SYCL2/SYCL1`;
     - split: `-sm tensor -ts 1/1`;
     - best validation used `-ub 128`;
     - decode: `42.106013 tok/s`, stddev `0.011783`;
     - total throughput: `75.570584 tok/s`;
   - three-B70 tensor split 512/512:
     - devices: `SYCL2/SYCL1/SYCL3`;
     - split: `-sm tensor -ts 1/1/1`;
     - `-ub 128`;
     - decode: `49.366188 tok/s`, stddev `0.486931`;
     - total throughput: `79.667255 tok/s`;
     - previous fused MMVQ2+SwiGLU 3x result: `46.804859 tok/s`;
     - LocalMaxxing ID: `cmoujcois00esld01c5s6bwht`;
   - four-B70 assist split:
     - selector `level_zero:2,1,3,0`;
     - split `1/1/1/0.05`;
     - failed before benchmark JSON with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during `MUL_MAT`;
     - log: `/home/steve/bench-results/qwen36-q4_0-gguf/rms-norm-mul-20260506/tensor4-assist005-sg2-p512n512/rmsmul-p512n512-r2-ub32-20260506T204412Z.log`;
     - decision: record as a failed diagnostic, do not submit to LocalMaxxing;
   - artifacts:
     - note: `/home/steve/llm-optimizations/notes/2026-05-06-q4-rmsnormmul.md`;
     - data: `/home/steve/llm-optimizations/data/qwen36-q4-rmsnormmul-20260506.json`;
   - decision:
     - keep `GGML_SYCL_FUSE_RMS_NORM_MUL=1` in the Q4_0 best stack;
     - current quality-preserving Q4_0 GGUF best is now 3x B70 at `49.366188 tok/s`;
     - next source-level target should be fused output-projection plus allreduce/residual epilogues or a broader same-activation GEMV group, because single-B70 is still short of the Windows Q4_0 target and 4x remains bottlenecked by narrow shards/collectives.
65. 2026-05-06 fused allreduce max-byte and FP8 PP2 post-reboot checks:
   - added `GGML_META_FUSE_ALLREDUCE_MAX_BYTES` as an opt-in ceiling for existing meta fused allreduce paths;
   - default behavior remains `64 KiB`;
   - graph probe with `GGML_META_ALLREDUCE_STATS=4` showed:
     - default ceiling fuses nearly all decode-sized projection residuals but no prompt-sized projection residuals;
     - `GGML_META_FUSE_ALLREDUCE_MAX_BYTES=1048576` fuses prompt `linear_attn_out`, `attn_output`, and `ffn_out` residual patterns in the `p32/n1` probe;
     - recurring miss is final-layer `attn_output -> GET_ROWS`;
   - performance:
     - prompt-only `p512/n1` default: `195.625197 tok/s`;
     - prompt-only `p512/n1` with `GGML_META_FUSE_ALLREDUCE_MAX_BYTES=16777216`: `196.233518 tok/s`;
     - full `p512/n512` with `GGML_META_FUSE_ALLREDUCE_MAX_BYTES=16777216`: prompt `188.183896 tok/s`, decode `49.346817 tok/s`;
     - conclusion: knob works but is not a validated speed improvement, so do not submit to LocalMaxxing;
   - source implication:
     - existing fused allreduce+ADD already covers the normal decode projection residual boundary;
     - next Q4_0 speed work should go below the meta boundary toward a projection GEMV plus reduction/residual epilogue, or target final-layer `GET_ROWS` if it becomes measurable;
   - FP8 post-reboot validation:
     - 4-rank XCCL gate passed on `level_zero:0,1,2,3`;
     - short PP2xTP2 non-spec FP8 load/generate passed at `512/32`, average latency `22.95836220899946 s`;
     - short PP2xTP2 CPU n-gram run completed at average latency `20.274570381006924 s` with no old GDN assertion or device-lost failure;
     - logs still show zero effective draft tokens (`draft=[0]`, `draft_lens=[0]`, `spec_lens={}`), so PP2 speculation remains a plumbing path, not a speed path;
   - artifacts:
     - `/home/steve/llm-optimizations/notes/2026-05-06-q4-allreduce-max-bytes.md`;
     - `/home/steve/llm-optimizations/notes/2026-05-06-fp8-pp2-postreboot-validation.md`;
     - `/home/steve/llm-optimizations/data/qwen36-q4-allreduce-max-bytes-20260506.json`;
     - `/home/steve/llm-optimizations/data/qwen36-fp8-pp2-postreboot-validation-20260506.json`;
     - `/home/steve/llm-optimizations/patches/llama-cpp-meta-allreduce-max-bytes-20260506.patch`.
66. 2026-05-06 current-stack allreduce + GET_ROWS recheck:
   - rechecked the existing off-by-default `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` hook after the Q8 cache, fused MMVQ2, fused MMVQ2+SwiGLU, RMS_NORM+MUL, event-barrier, and sync-after-2 stack was in place;
   - decode-only A/B:
     - gate off, `p0/n512/r3`: `48.628094 tok/s`;
     - gate on, `p0/n512/r3`: `49.043584 tok/s`;
   - full `512/512/r5` A/B:
     - gate off: prompt `196.860926 tok/s`, decode `48.827917 tok/s`, stddev `0.072730`;
     - gate on: prompt `197.252755 tok/s`, decode `49.403656 tok/s`, stddev `0.361676`;
     - computed total for the submitted run: `79.016858 tok/s`;
   - correctness:
     - `llama-completion -no-cnv`, greedy decode, same prompt and seed;
     - baseline and GET_ROWS stdout SHA256 both `2039492ece1be609e945c074396527ae6e0bcaddd2cf82cce6fd847355711214`;
     - same Q4_0 weights, same f16 KV, no speculative decoding, no sampling change, no power changes;
   - LocalMaxxing:
     - accepted ID `cmoultsa900h0ld011f0r2hcs`;
   - decision:
     - keep `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` in the current TP3 best recipe;
     - this is a narrow final-logits win, not the larger low-level projection epilogue we still need for a step-change;
     - next Q4_0 work remains projection GEMV plus allreduce/residual epilogue or same-activation multi-GEMV fusion.
67. 2026-05-06 projection epilogue scheduler diagnostic:
   - implemented an off-by-default `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=1` path that recognizes Q4_0 `MUL_MAT` partials followed by fused allreduce+ADD and routes them through a new SYCL backend helper;
   - first Q8-on smoke found a correctness/lifetime bug in the fallback shape:
     - meta had already removed the `MUL_MAT` from the normal backend graph before helper decline;
     - fallback then computed the skipped `MUL_MAT` through the aux-node path while `GGML_SYCL_Q8_CACHE=1`;
     - the run wrote a JSON result but aborted at teardown with `GGML_ASSERT(pool_size == 0) failed`;
   - fixed by adding a planner-level guard:
     - when `GGML_SYCL_Q8_CACHE` is nonzero, meta does not form this fusion at all;
     - SYCL helper also declines if Q8 cache is active;
   - validation:
     - Q8-on guarded `p0/n1`: no abort, zero `backend+mulmat+add` paths, GET_ROWS path remains active;
     - Q8-off `p0/n1`: 142 `backend+mulmat+add` path entries, 2 GET_ROWS entries, no assertions;
   - Q8-off short decode A/B, `p0/n128`:
     - gate off: `48.239722 tok/s`;
     - gate on: `47.700182 tok/s`;
     - delta: `-0.539540 tok/s` (`-1.12%`);
   - decision:
     - keep `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=0` in the current best recipe;
     - do not submit to LocalMaxxing because this was diagnostic-only, required Q8 disabled, and regressed short decode;
     - preserve the patch because it maps the correct graph boundary and proves the helper dispatch path, but next work should be lower-level: fuse MMVQ/reduction/residual epilogue or group same-activation projections before the collective;
   - artifacts:
     - note: `/home/steve/llm-optimizations/notes/2026-05-06-q4-projection-epilogue-diagnostic.md`;
     - data: `/home/steve/llm-optimizations/data/qwen36-q4-projection-epilogue-diagnostic-20260506.json`;
     - patch: `/home/steve/llm-optimizations/patches/llama-cpp-sycl-meta-mulmat-add-diagnostic-current-20260506.patch.gz.b64`.
68. 2026-05-06 current-stack single-B70 subgroup runtime screen:
   - tested `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME` after the Q8 cache, fused MMVQ2, fused MMVQ2+SwiGLU, and RMS_NORM+MUL stack;
   - command shape: `SYCL2`, `-sm none`, `-fa 1`, `-ub 128`, `-ctk f16`, `-ctv f16`, `p0/n256/r2`;
   - results:
     - default: `24.930018 tok/s`;
     - `1`: `24.894307 tok/s`;
     - `16`: `24.893128 tok/s`;
     - `2`: `24.886190 tok/s`;
     - `8`: `24.876417 tok/s`;
     - `4`: `24.874003 tok/s`;
   - decision:
     - default remains best;
     - do not set `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME` in the current single-B70 recipe;
     - this closes another simple MMVQ runtime tuning branch; reaching the Windows `>=27 tok/s` single-B70 target likely requires deeper Q4_0 matvec work rather than runtime subgroup count changes;
   - artifacts:
     - note: `/home/steve/llm-optimizations/notes/2026-05-06-q4-single-subgroup-current-negative.md`;
     - data: `/home/steve/llm-optimizations/data/qwen36-q4-single-subgroup-current-20260506.json`;
     - TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/single-subgroup-current-20260506/single-subgroup-current-p0n256-r2-20260506T225012Z.tsv`.

## Success Criteria

- First accepted improvement: Q4_0 GGUF single B70 `>=27 tok/s` with a reproducible command and no quality-changing flags.
- Strong success: Q4_0 GGUF single B70 `>=29 tok/s`.
- FP8 investigation success: official `Qwen/Qwen3.6-27B-FP8` reaches or beats the current Q4_0 TP3 result while preserving output quality.
- Static FP8 current best: `vrfai/Qwen3.6-27B-FP8` on four B70s with patched vLLM/XPU FlashAttention2 plus n-gram speculative decode reaches `49.581893 tok/s` on 512 prompt / 512 output. This is ahead of the Q4_0 TP3 validation while preserving more fidelity than INT4 paths.
- Static FP8 32k-context status: TP4 with patched vLLM/XPU FlashAttention2 succeeds at `max_model_len=32768` and reports `1,133,163` GPU KV-cache tokens with `42.996276 tok/s` at 2048 prompt / 256 output. TP2/PP2 also fits but is slower (`26.361533 tok/s`) and reports slightly less 32k KV capacity.
- Current dual-card milestone reached: Q4_0 GGUF single session `42.106013 tok/s` on two B70s for 512 prompt / 512 output with the current fused stack, quality preserving, software-only.
- Current improved multi-card milestone: Q4_0 GGUF single session `49.403656 tok/s` on three B70s for 512 prompt / 512 output with Q8 activation cache, fused MMVQ2, fused MMVQ2+SwiGLU, fused RMS_NORM+scale-MUL, fused allreduce+GET_ROWS, single-kernel allreduce, fused allreduce+ADD, `GGML_SYCL_COMM_SYNC_AFTER=2`, and `-ub 128`. This is quality preserving and software-only.
- Current four-card Q4_0 status: assist split `1/1/1/0.05` reaches `39.204149 tok/s`, improving the equal-split four-card result by `12.24%` but still trailing the best three-card result, so quad Q4_0 remains a kernel/scheduling investigation rather than the production path. The RMS_NORM+MUL rerun of that assist split failed with Level Zero OOM during `MUL_MAT`.
- Dual-card success: Q4_0 GGUF single session `>=48 tok/s` first, then `>=52 tok/s`, without switching away from Q4_0.
- Four-card success: Q4_0 GGUF single session must exceed dual-card by a meaningful margin before treating quad tensor split as viable.
