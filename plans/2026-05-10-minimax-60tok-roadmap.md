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

These targets should move upward with quantization/runtime changes. The
AutoRound INT4 path is materially faster than the GGUF capacity path, so the
old `>30 tok/s` starting target is now just the minimum bar. A useful B70
software result should either move the current quality-cleared p512/n1536
reference by at least a few percent, or explain a bottleneck that blocks the
`60 tok/s` target.

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
   - Plain provider swapping is not enough: installed `+rms_norm` and
     source-tree `fused_add_rms_norm=["xpu_kernels"]` both stayed below the
     installed-runtime reference. Future work needs to fuse the collective
     boundary with adjacent epilogue work, not just replace the RMSNorm kernel.
   - An installed-runtime post-attention `fused_add_rms_norm` screen also
     failed to beat the reference: warm p512/n512 reached `35.077` output tok/s
     alone and `35.804` when paired with delayed `o_proj` allreduce. Do not
     spend more time on provider swaps without collective fusion.
   - A Python-level custom op that wraps `dist.all_reduce` plus
     `_C.fused_add_rms_norm` warmed to only `32.611` output tok/s. The fusion
     must happen in C++/SYCL or as a real compiler/kernel pass; Python dispatch
     and extra clones are too expensive.
   - A guarded XPU out-of-place functional allreduce experiment removed the
     clone/copy pairs around all 187 TP collectives, but warmed p512/n512 only
     reached `35.64` output tok/s versus the accepted `39.610585` reference.
     Cleaner functional allreduce tracing is not enough; keep this closed as a
     negative unless paired with a real fused epilogue.
   - An env-gated XPU `torch.ops.vllm.all_reduce` custom-op collective screen
     also changed the graph shape, replacing `_c10d_functional.all_reduce` and
     `wait_tensor` call sites with opaque `vllm.all_reduce` sites. It still
     warmed to only `34.980` output tok/s at p512/n512, and emitted PyTorch
     aliasing warnings because the Python op returns an input alias. Treat this
     as evidence that the next useful collective path must be C++/SYCL or
     backend-level, not another Python/opaque wrapper.

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
   - A fast-NVMe DFlash retest loaded and compiled both target and drafter, but
     still stalled before generating a p64/n32 sample. Storage was not the
     blocker.
   - Revisit speculation only with a draft that has target-model verification and a measured acceptance rate high enough to beat the extra scheduler and KV pressure.
   - If speculation works, publish both decode tok/s and total/prefill tok/s to LocalMaxxing with the spec method and draft model recorded.

7. Secondary dense-model comparisons
   - Qwen3.6/Qwen3.5 27B/35B FP8 or Q4 can still be useful to isolate dense attention/TP behavior from MiniMax MoE behavior.
   - These are secondary probes, not the main objective, unless they reveal a transferable XPU communication or graph-scheduling fix.

## External Reference Points

- vLLM documents speculative decoding methods including `ngram`, `suffix`,
  `mtp`, `eagle3`, and `dflash`, and describes the framework as target
  verified. That keeps speculation on the roadmap, but only results that
  preserve target-model verification should count as quality-preserving
  MiniMax wins.
- vLLM's Intel AutoRound page lists W4A16 and W8A16 as the currently enabled
  Intel-platform recipes. Our MiniMax AutoRound path is therefore aligned with
  the actively supported Intel quantization surface, even though the B70/XPU
  MoE fast path is still local and experimental.
- Intel `llm-scaler` now advertises Arc Pro B60/B70 support and its vLLM image
  calls out CCL P2P/USM, INT4/FP8 serving, tensor/pipeline/data parallelism,
  and context-length tooling. We should continue mining it for kernel and
  scheduling ideas, but local B70 measurements still decide whether a path is
  useful.
- vLLM's current fusion documentation lists MiniMax QK Norm as a
  MiniMax-specific pass for Q/K variance allreduce plus RMSNorm, off by default
  and currently requiring the CUDA `minimax_allreduce_rms_qk` custom op. That
  validates our local direction: port or replace this boundary for XPU rather
  than enabling CUDA/ROCm fusion passes globally.
- Public MiniMax M2.7 hardware reports for non-Intel systems are higher than
  our current B70 result: one recent 32k-context llama.cpp report lists about
  `71.52 tok/s` on 4x RTX 4090, `118.74 tok/s` on one RTX PRO 6000, and
  `24.41 tok/s` on DGX Spark. The quantization and backend differ from our
  AutoRound/vLLM setup, but these numbers support setting the B70 aspiration
  above `60 tok/s` once TP communication and graph scheduling improve.
- DFlash remains technically interesting because the paper claims lossless
  acceleration from block-diffusion drafting, but the current MiniMax B70
  DFlash harness loads and compiles then stalls before producing a throughput
  result. Do not count it until the benchmark completes and reports acceptance
  behavior.
- SGLang's current expert-parallel documentation points fast MoE serving toward
  specialized routed backends such as DeepEP/DeepGEMM and FlashInfer/TRT-LLM
  routed MoE kernels. These are NVIDIA-specific today, but the design lesson is
  relevant: MiniMax speed comes from routed-MoE/communication-aware kernels, not
  from generic per-layer TP collectives.
- KTransformers' public benchmark board shows strong single-session decode for
  dense and MoE models on RTX 5090 systems, and a public MiniMax 2.5 8x-3090
  report used SGLang TP/EP rather than simple pipeline mode. For B70 this argues
  against spending more time on TP2/PP2 for batch-1 latency, and for investing
  in an XPU equivalent of routed MoE plus collective epilogue fusion.
- REAP-pruned MiniMax M2.7 AutoRound W4A16 variants exist and may be useful as
  a separate quality/speed tradeoff track, but they reduce experts and are not a
  drop-in quality-preserving improvement for the current full MiniMax AutoRound
  target.

## Immediate Next Tests

1. Prototype an XPU-specific fused allreduce/residual/RMSNorm boundary as a default-off patch. The standalone RMS provider and delayed-allreduce screens are closed as negative.
2. Build a lower-overhead decode timing pass around attention, Q/K RMS, projections, and MoE that can run for short diagnostics without perturbing real throughput.
3. Inspect compiled IR around hidden-state allreduce wait sites and target the first boundary that feeds immediately into residual/RMS or MoE epilogue work.
4. Re-run p512/n512 and p512/n1536 after each code change, then submit only quality-preserving, repeatable improvements to LocalMaxxing.

Current code-direction preference after the latest negative screens:

1. Keep the active runtime on the clean quality path for baseline runs.
2. Avoid Python custom-op wrappers around `dist.all_reduce`; they add clones and
   dispatch overhead without changing the real boundary.
3. Implement the next fusion in C++/SYCL or a real compiler pass, starting with
   the smallest MiniMax-specific shape: TP4, decode-sized tensors, Q/K RMS or
   hidden-state allreduce followed immediately by residual add/RMSNorm.
4. If a fusion cannot beat the current p512/n1536 anchor, archive it as a
   negative and leave the env flag unset.
