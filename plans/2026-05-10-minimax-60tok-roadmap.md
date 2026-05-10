# MiniMax M2.7 AutoRound 60 tok/s Roadmap, 2026-05-10

## Objective

The MiniMax M2.7 AutoRound INT4 path is now the primary four-B70 optimization target. The old `>30 tok/s` and `>40 tok/s` goals were appropriate while proving that the model could run correctly across four cards, but they are no longer ambitious enough for the AutoRound/llm-scaler path.

Current accepted reference points:

- Conservative long-run anchor: `37.552538` output tok/s and `50.070051` total tok/s at p512/n1536, TP4, no speculation, no expert dropping, no power-limit change, Q/K TP variance allreduce enabled. LocalMaxxing: `cmozow03v005wlo01q81bnspx`.
- Accepted short/mid speed points: `39.610585` output tok/s at p512/n512 and `40.303730` output tok/s at p512/n1024 using the fast-NVMe FP16 u4 decode recipe.
- The earlier `41.130667` p512/n1536 result remains useful as a scheduling clue, but it is not the quality-cleared target because the cached AOT graph did not visibly include the per-layer Q/K RMS variance allreduce.

Revised targets:

- Near-term repeatable target: `45 tok/s` output at p512/n1536 without changing model quality.
- Main target: `60 tok/s` output at p512/n1536 on 4x B70 with the MiniMax AutoRound INT4 model.
- Stretch target: `75+ tok/s` only if achieved by verified speculative decoding, MTP-style target-compatible drafting, or deeper source-level fusion that preserves target logits.

## Quality Guardrails

Only promote results that preserve target-model behavior:

- Do not skip Q/K RMS TP variance allreduce.
- Do not drop experts or silently route missing experts.
- Do not use root-residual or deferred-reduction paths that cross RMSNorm, router, MoE, or attention nonlinearities unless token/logit checks prove equivalence.
- Do not count lower-quality quantization changes as MiniMax AutoRound speed improvements unless they are labeled separately.
- Do not count speculative decoding unless the target model verifies accepted tokens.
- Keep benchmark payloads explicit about prompt tokens, output tokens, context window, total tok/s, and output tok/s.

## Current Bottleneck Model

The u4 MoE bridge is no longer the only ceiling. Existing timing notes put MiniMax decode work roughly in these buckets:

- Q/K RMS plus TP collectives: meaningful per-layer cost and many graph boundaries.
- Attention/KV and projection scheduling: likely exposed after the u4 MoE speedup.
- MoE decode kernels: still significant, but standalone htile and router-logits experiments did not improve full vLLM throughput.
- vLLM compiled graph and AOT cache behavior: cold isolated caches can produce the 9,408-KV artifact and lose 25-30% throughput until a favorable cache is reused.
- oneCCL itself is fast for the tiny allreduce sizes in isolation; the bigger issue is where collectives sit in the graph and how much fencing/cloning/scheduling they force.

## Workstreams

1. XPU allreduce plus residual/RMSNorm fusion
   - Stock vLLM `fuse_allreduce_rms` is not useful for this stack today: XPU disables it, and the current implementation is CUDA/FlashInfer/ROCm oriented.
   - Build a B70/XPU-specific path around Level Zero/XCCL-visible tensors instead of trying to enable the CUDA pass.
   - Start with post-attention and post-MoE boundaries where the hidden-state allreduce feeds residual add and RMSNorm.

2. Q/K RMS variance fusion
   - Standalone helper kernels and standalone mailbox allreduce were negative.
   - Revisit Q/K only as a larger fused kernel that includes useful adjacent work, such as qkv reshape, variance, cross-rank reduction, RMS application, and possibly RoPE.
   - Any custom op must register before AOT graph load, otherwise cached graphs fail before the benchmark starts.

3. Attention/KV profiling
   - Avoid live `xpu-smi stats` polling during real benchmarks; it stalled vLLM shared-memory synchronization.
   - Prefer lower-overhead Level Zero events, SYCL event timing, or short synchronized diagnostic runs.
   - Measure whether KV writes, attention kernels, or projection kernels now dominate after the u4 MoE bridge.

4. MoE epilogue fusion
   - Delaying MoE allreduce and output-projection allreduce gave small short-run improvements but did not repeat above the conservative long-run reference.
   - The useful direction is not just moving collectives; it is combining the u4 expert output, output epilogue, allreduce, residual add, and RMSNorm where graph-safe.

5. Scheduler and graph-cache hygiene
   - Isolate experimental `VLLM_CACHE_ROOT` directories.
   - Always record cold versus warm behavior and GPU KV-cache tokens.
   - Treat the `9,408` KV-token artifact as a performance smell, not a valid speed floor.
   - Keep dormant helper/timing branches out of active runtime unless the corresponding env flag is being tested.

6. Speculative decode and MTP
   - Current MiniMax n-gram, ngram_gpu, and DFlash screens are negative or unstable.
   - Revisit speculation only with a draft that has target-model verification and a measured acceptance rate high enough to beat the extra scheduler and KV pressure.
   - If speculation works, publish both decode tok/s and total/prefill tok/s to LocalMaxxing with the spec method and draft model recorded.

7. Secondary dense-model comparisons
   - Qwen3.6/Qwen3.5 27B/35B FP8 or Q4 can still be useful to isolate dense attention/TP behavior from MiniMax MoE behavior.
   - These are secondary probes, not the main objective, unless they reveal a transferable XPU communication or graph-scheduling fix.

## Immediate Next Tests

1. Screen oneCCL worker affinity with default worker count. Do not increase `CCL_WORKER_COUNT` because `CCL_WORKER_COUNT=2` hung during XCCL initialization and Intel documents that multiple workers are not recommended for GPU buffers.
2. Build a lower-overhead decode timing pass around attention, Q/K RMS, projections, and MoE that can run for short diagnostics without perturbing real throughput.
3. Prototype an XPU-specific fused allreduce/residual/RMSNorm boundary as a default-off patch.
4. Re-run p512/n512 and p512/n1536 after each code change, then submit only quality-preserving, repeatable improvements to LocalMaxxing.
