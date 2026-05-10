# Intel Arc Pro B70 LLM Lab Notes

Date: 2026-05-04

This note is the current public/reproducible summary for the B70 optimization work. The active technical plan is `plans/q4_0-gguf-b70-optimization-plan.md`; submitted benchmark IDs and exact payloads are recorded in `notes/localmaxxing-submissions-2026-05-04.md` and `data/localmaxxing-payloads-20260504.json.gz.b64`.

## 2026-05-10 MiniMax AutoRound Addendum

MiniMax M2.7 AutoRound INT4 is now the main four-card optimization target. The
aspiration target has been raised to `60 tok/s` output at p512/n1536 on 4x B70,
with `75+ tok/s` reserved for verified speculative/MTP or deeper fusion that
preserves target logits.

Current quality-conservative anchor:

- `37.552538` output tok/s / `50.070051` total tok/s at p512/n1536, TP4,
  FP16, llm-scaler raw-u4 decode MoE path, Q/K TP variance allreduce enabled,
  no speculation, no expert dropping, and no power-limit change. LocalMaxxing:
  `cmozow03v005wlo01q81bnspx`.

Recent negative follow-ups:

- DFlash from fast NVMe loads/compiles target and drafter but stalls before
  producing a p64/n32 benchmark result.
- Source-tree and installed-runtime RMS provider swaps are below the accepted
  MiniMax reference.
- Installed-runtime post-attention fused-add RMS warmed to `35.077` output
  tok/s at p512/n512, and delayed `o_proj` allreduce plus fused-add RMS warmed
  to `35.804`. Both are negative versus the accepted `39.611` p512/n512
  reference and were not submitted to LocalMaxxing.
- A Python-level custom op wrapping output-projection allreduce plus
  `_C.fused_add_rms_norm` compiled after moving registration out of the forward
  path, but warmed to only `32.611` output tok/s at p512/n512. This rules out
  Python custom-op wrapping as the practical fusion layer.

Current direction:

- Stop spending time on standalone RMS provider swaps or simply moving the same
  allreduce call.
- Build an XPU-specific allreduce plus residual/RMSNorm or MoE/projection
  epilogue fusion path, with p512/n512 and p512/n1536 validation after each
  change.
- Keep all negative/diagnostic flags unset for real benchmarks unless a run is
  explicitly labeled as an experiment.

## Hardware And Constraints

- Host: Ubuntu 24.04.4 LTS, kernel 6.17.0-22-generic during the latest runs.
- CPU: AMD EPYC 9015.
- GPUs: 4x Intel Arc Pro B70 32GB, exposed through Level Zero selectors `0-3`.
- Power-limit / overclocking changes are intentionally out of scope. The current work is software-only.
- Single-card tests should isolate one device with `ONEAPI_DEVICE_SELECTOR=level_zero:N`.
- llama.cpp multi-GPU device syntax is slash-separated, for example `-dev SYCL0/SYCL1/SYCL2`.

## Current Best Results

Quality-preserving Q4_0 GGUF path, llama.cpp/SYCL:

- 1x B70 baseline: about `24.7 tok/s` decode.
- 2x B70 async-copy tensor split: `37.690 tok/s`, LocalMaxxing `cmoqkcqpv0006la04l5mtlj2q`.
- 2x B70 single-kernel allreduce: `39.849 tok/s`, LocalMaxxing `cmoqp6jpq0004lb04241n9ns3`.
- 2x B70 Q8 activation-cache validation, 512 prompt / 512 output: `40.487 tok/s`, LocalMaxxing `cmormylxz000fib04wodwo1ng`.
- 3x B70 single-kernel allreduce, selector/root order `2,1,3`: `41.737 tok/s`, LocalMaxxing `cmoqqed6s0007jv049wnizwle`.
- 3x B70 Q8 activation-cache short run, 512 prompt / 256 output: `42.432 tok/s`, LocalMaxxing `cmordq9t5000dl404x309pj48`.
- 3x B70 Q8 activation-cache validation, 512 prompt / 512 output: `41.659 tok/s`, LocalMaxxing `cmorn71e2000kib0415vo51vj`.
- 3x B70 current quality-cleared no-root Q4_0 stack with experimental flat fused Qwen35 `ssm_ba` GGUF, 512 prompt / 512 output: `50.130 tok/s`, LocalMaxxing `cmov6p4r7007tqr01yi8ug4un`.
- 3x B70 root-residual performance ceiling with `--poll 25`, 512 prompt / 512 output: `50.922 tok/s`, LocalMaxxing `cmouxjqao000npn01hxqn68td`, now marked suspect because later token/logit probing found the root-residual plus meta allreduce-add pair can diverge.
- 3x B70 final-rebuild root-residual rerun with flat fused `ssm_ba` GGUF: `50.687 tok/s` decode and `80.879 tok/s` total. Default-prompt root checks passed, but a two-token prompt follow-up timed out, so this is documented but not submitted/promoted.
- 4x B70 Q4_0 remains negative-scaling: `31.482 tok/s` without Q8 cache and `31.913 tok/s` with Q8 cache; LocalMaxxing `cmor2e5r00004jl04o99d26p8` and `cmornec37000okw040zl9563z`.

FP8 path, vLLM/XPU:

- Official `Qwen/Qwen3.6-27B-FP8` runs, but current XPU block-FP8/requant path is slow: TP2 512-output upper-bound `20.106 tok/s`, LocalMaxxing `cmorb75xb001ckz0489eqc9se`.
- Static `vrfai/Qwen3.6-27B-FP8` with patched XPU FlashAttention2 reaches `41.503 tok/s` on TP4 for 512 prompt / 512 output, LocalMaxxing `cmork3n3k000ujo04y73lbr1j`.
- TP4 also fits Qwen3.6 full configured context (`262144`) and reports `1,206,355` GPU KV-cache tokens.
- PP2 x TP2 is valid as a capacity fallback but slower for one sequence: `22.721 tok/s`, LocalMaxxing `cmormmlz0000bky04wpu4oc01`.
- FP8 KV cache is not a speed path and is quality-risky without proper scaling: `28.036 tok/s`, LocalMaxxing `cmornlh8g000vkw04yb57ukvl`.

MiniMax M2.7 UD-IQ4_XS path:

- First useful four-B70 MiniMax baseline: `13.754 tok/s` for `p0/n64` with `ik_llama.cpp` process-per-GPU RPC workers, SYCL/Level Zero, layer split, runtime repack, CPU KV, fused MoE off, fused MMAD off, and local SYCL `MULTI_ADD`. LocalMaxxing `cmovvoo6f00f5p1017yeb7kxd`.
- Current MiniMax best: `16.384 tok/s` for `p0/n64/r3` after the corrected RPC device map and `-nkvo 0`, with conservative SYCL `MOE_FUSED_UP_GATE`, fused MoE, merged gate/up experts (`-muge 1`), and experimental SYCL `MUL_MULTI_ADD`. LocalMaxxing `cmowft2hr000oo3019is4snoq`.
- Direct single-process SYCL MiniMax is blocked on a regular SYCL model-buffer allocation during `llm_load_tensors`. Even an uneven split plus `-b 512` fails on a 19.028 GB allocation on GPU0 despite full reported VRAM. The process-per-GPU RPC layout remains the valid path until regular model buffers can be chunked or routed through the split/pool allocator.
- Layer placement is only a small/noisy lever: one-repeat `p0/n64` sweep topped out at `16.358 tok/s` with `-ts 0.8/1.05/1.05/1.1`, below the existing `16.384 tok/s` three-repeat best.
- Quality-correct MiniMax graph mode now executes with forced real reductions at nonlinear boundaries, but it is diagnostic only: `GGML_MINIMAX_NO_DEFER_REDUCE=1` plus `GGML_RPC_REDUCE_MIRROR=1` reached only `2.034 tok/s` for a one-token smoke. The earlier faster branch-fused graph path remains unpromoted because deferred reductions can cross RMSNorm/router/MoE and change the math.
- Layer-mode follow-up screens were negative: `-t` sweep topped out at `16.307 tok/s`, `-fa 1` aborts on unsupported `FLASH_ATTN_EXT`, disabling fused MMAD/MoE is slower at p0/n64, oneDNN enabled is slower at `15.590 tok/s`, same-type contiguous copy memcpy is neutral, and an 8-expert `MUL_MULTI_ADD` unroll regressed to `13.823 tok/s` and was removed.
- CPY tracing shows MiniMax repeats three copy shapes per layer: f32-to-f32 row-strided, contiguous f32-to-f16, and `ne0=1` strided f32-to-f16. A default-off standalone shape-specific copy fast path regressed to `12.732 tok/s`, so the next copy-related attempt should fuse producer kernels into KV/cache writes instead of replacing `CPY` with separate kernels.
- Fused RMSNorm is no longer an unsupported-op blocker in the local SYCL RPC worker. The f32 fused RMSNorm implementation runs, but p0/n64/r1 reached `16.308 tok/s`, below the current `16.384 tok/s` best.

INT4 AutoRound path:

- `Lorbus/Qwen3.6-27B-int4-AutoRound` produced strong vLLM/XPU speed results, including `45.2 tok/s` on 1x B70 and `49.1 tok/s` on 2x B70.
- These results are recorded on LocalMaxxing but are not counted as Q4_0 GGUF success because the quantization changes model fidelity.

## Important Implementation Artifacts

llama.cpp Q4_0/SYCL work:

- Combined diff: `patches/llama-cpp-db44417-b70-sycl-combined.diff.gz.b64`.
- Decode/apply guide: `patches/llama-cpp-db44417-b70-sycl-combined-diff.md`.
- Key runtime flags for the best Q4_0 runs:
  - `GGML_SYCL_ASYNC_CPY_TENSOR=1`
  - `GGML_SYCL_COMM_ALLREDUCE=1`
  - `GGML_SYCL_COMM_SINGLE_KERNEL=1`
  - `GGML_SYCL_Q8_CACHE=1`
- Benchmark harnesses:
  - `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`
  - `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`

vLLM/XPU FP8 work:

- XPU FA2 singleton scale patch: `patches/vllm-xpu-fa2-compressed-tensors-scalar-scales.patch`.
- Qwen3.5/Qwen3.6 language-only vision skip patch: `patches/vllm-qwen35-language-model-only-skip-vision.patch`.
- FP8 result notes: `notes/2026-05-04-qwen36-fp8-b70-fa2.md` and `notes/2026-05-04-qwen36-fp8-full-context-topologies.md`.
- FP8 topology data: `data/qwen36-fp8-b70-topology-screens-20260504.json`.

## Current Diagnosis

- Single-card Q4_0 is not limited by flash attention, ubatch, graph capture, oneDNN, AOT alone, or a missing reordered MMVQ path.
- Reordered Q4_0 MMVQ is required; disabling it drops single-card speed to about `15 tok/s`.
- Multi-card Q4_0 improves through async tensor copies, direct allreduce, Q8 activation caching, graph fusions, fused small projections, and safe allreduce+ADD scheduling. Root-residual fused allreduce+ADD is a promising performance ceiling but is not quality-cleared until its ordering hazard with meta allreduce-add is fixed. Four-card scaling still regresses because each token pays many small 20 KiB reductions and narrower row shards lose matvec efficiency.
- Timing hooks show synchronized allreduce cost rises sharply with GPU count: roughly `1.718 ms/token` on 2 GPUs, `5.732 ms/token` on 3 GPUs, and `10.605 ms/token` on 4 GPUs for the same reduction pattern.
- Four-GPU root/order/topology sweeps are not enough; the next useful work is reducing the number of reductions or fusing delayed reductions through safe graph regions.
- MiniMax M2.7 is now past the pure-capacity stage by using one RPC process per B70. The working baseline is 16.384 tok/s. The quality-correct graph diagnostic shows real reduce/broadcast must happen before nonlinear boundaries, and the current client-side RPC implementation is too slow. Simple runtime knobs did not move the layer path. The >30 tok/s target likely requires shape-specific layer kernels, implementing missing SYCL worker ops, or designing graph/tensor parallelism around a much cheaper collective.

## Current Next Steps

1. Keep Q4_0 single-card profiling focused on reordered MMVQ and activation quantization launch overhead.
2. Prototype fewer/fused Meta allreduces for Q4_0 multi-GPU; do not spend more time on simple root-copy or pairwise allreduce topology variants.
3. Use 3x B70 Q4_0 no-root fused beta/alpha at `50.130 tok/s` as the current quality-cleared speed point; treat root-residual `50.922 tok/s` as a performance ceiling until the correctness hazard is fixed.
4. Use static FP8 TP4 with patched XPU FA2 as the best high-fidelity four-card path for now.
5. Keep PP2 x TP2 as a capacity fallback for larger models, not a speed path for Qwen3.6 27B.
6. Mine Intel `llm-scaler` for reduce-scatter/all-gather, fused output-kernel, Gated DeltaNet, and speculative/MTP ideas, but do not assume it will run directly on Arc/B70.
7. For MiniMax, keep the process-per-GPU RPC layout as the capacity baseline while turning the newly working SYCL fused-MoE path into a lower-level active-expert kernel. Treat naive graph split as diagnostic until reduce/broadcast can move off the host-mediated RPC path. Direct SYCL needs chunked regular model-buffer allocation before it is usable.
