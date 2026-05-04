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

## Success Criteria

- First accepted improvement: Q4_0 GGUF single B70 `>=27 tok/s` with a reproducible command and no quality-changing flags.
- Strong success: Q4_0 GGUF single B70 `>=29 tok/s`.
- FP8 investigation success: official `Qwen/Qwen3.6-27B-FP8` reaches or beats the current Q4_0 TP3 result while preserving output quality.
- Static FP8 current best: `vrfai/Qwen3.6-27B-FP8` on four B70s with patched vLLM/XPU FlashAttention2 reaches `41.503 tok/s` on 512 prompt / 512 output. This is effectively tied with the Q4_0 TP3 validation while preserving more fidelity than INT4 paths.
- Static FP8 full-context status: TP4 with patched vLLM/XPU FlashAttention2 succeeds at Qwen3.6's configured `262,144` token context and reports `1,206,355` GPU KV-cache tokens. PP2 x TP2 also fits but is slower for a single sequence.
- Current dual-card milestone reached: Q4_0 GGUF single session `37.690 tok/s`, quality preserving, software-only.
- Current improved multi-card milestone: Q4_0 GGUF single session `42.432 tok/s` on three B70s for 512 prompt / 256 output with `GGML_SYCL_Q8_CACHE=1`; longer 512-output validation is `41.659 tok/s`. This is quality preserving and software-only.
- Dual-card success: Q4_0 GGUF single session `>=48 tok/s` first, then `>=52 tok/s`, without switching away from Q4_0.
- Four-card success: Q4_0 GGUF single session must exceed dual-card by a meaningful margin before treating quad tensor split as viable.
