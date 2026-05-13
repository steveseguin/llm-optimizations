# MiniMax M2.7 AutoRound 60 tok/s Roadmap, 2026-05-10

## Objective

The MiniMax M2.7 AutoRound INT4 path is now the primary four-B70 optimization target. The old `>30 tok/s` and `>40 tok/s` goals were appropriate while proving that the model could run correctly across four cards, but they are no longer ambitious enough for the AutoRound/llm-scaler path.

Current accepted reference points:

- Conservative long-run anchor: `37.552538` output tok/s and `50.070051` total tok/s at p512/n1536, TP4, no speculation, no expert dropping, no power-limit change, Q/K TP variance allreduce enabled. LocalMaxxing: `cmozow03v005wlo01q81bnspx`.
- Current post-CCL-cleanup quality floor: `38.046755` output tok/s and
  `50.729007` total tok/s at p512/n1536 after restoring oneCCL ATL local-rank
  inference as the default path. The top-level compiled graph contains the
  `f32[s72,2]` Q/K variance allreduce signature. LocalMaxxing:
  `cmp27nihp001orm01dataqtfq`.
- Previous graph-partition-only quality-cleared long-run result: `40.209882` output tok/s and
  `53.613176` total tok/s at p512/n1536 with
  `--compilation-config '{"use_inductor_graph_partition":true}'`, AOT
  `c3f2b10098683775b74b9bb91c9a44570f4df792c7a1b0061b5df73b6ef18f20`, and
  the Q/K variance allreduce preserved. LocalMaxxing:
  `cmp2kd5ux006frm013il4qu13`.
- Current best quality-cleared long-run result: `48.092807` output tok/s and
  `64.123742` total tok/s at p512/n1536 with the same static decode graph plus
  the vLLM benchmark async engine. It uses
  `--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1]}'`,
  AOT `3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`,
  `--async-engine`, and the Q/K variance allreduce preserved. LocalMaxxing:
  `cmp3cgooj0019s401d7p1ks3e`.
- Previous non-async static decode long-run result: `47.586110` output tok/s
  and `63.448146` total tok/s at p512/n1536 with the same AOT graph and Q/K
  variance allreduce preserved. LocalMaxxing: `cmp2mf1zw007wrm01op7aimhk`.
- Accepted short/mid speed points: `39.610585` output tok/s at p512/n512 and `40.303730` output tok/s at p512/n1024 using the fast-NVMe FP16 u4 decode recipe.
- The earlier `41.130667` p512/n1536 result remains useful as a scheduling clue, but it is not the quality-cleared target because the cached AOT graph did not visibly include the per-layer Q/K RMS variance allreduce.

Revised targets:

- Near-term repeatable target: `50 tok/s` output at p512/n1536 without changing model quality.
- Main target: `60 tok/s` output at p512/n1536 on 4x B70 with the MiniMax AutoRound INT4 model.
- First stretch target: `75+ tok/s` only if achieved by verified speculative decoding, MTP-style target-compatible drafting, or deeper source-level fusion that preserves target logits.
- Secondary stretch target: `90+ tok/s` if speculative acceptance is high enough and the run records target verification, acceptance behavior, total tok/s, output tok/s, and TTFT/prefill data where available.

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
   - A C++ `CompositeExplicitAutograd` op for post-attention allreduce plus
     residual add plus RMSNorm compiled and executed after passing
     `get_tp_group().device_group.group_name` instead of vLLM's logical
     `tp:0` name. It inserted `62` fused call sites into the compiled graph,
     but p512/n512 regressed to `19.89` output tok/s and `39.78` total tok/s.
     The boundary is still important, but an opaque wrapper around c10d plus
     ATen math is not enough. The next attempt must own a real SYCL epilogue or
     compiler lowering.

2. Q/K RMS variance fusion
   - Standalone helper kernels and standalone mailbox allreduce were negative.
   - Revisit Q/K only as a larger fused kernel that includes useful adjacent work, such as qkv reshape, variance, cross-rank reduction, RMS application, and possibly RoPE.
   - Any custom op must register before AOT graph load, otherwise cached graphs fail before the benchmark starts.

3. Attention/KV profiling
   - Avoid live `xpu-smi stats` polling during real benchmarks; it stalled vLLM shared-memory synchronization.
   - Avoid synchronized full-model decode timing during real benchmarks as
     well. A p512/n128 run with `VLLM_XPU_DECODE_TIMING_SYNC=1` stalled before
     normal model-load progress. Use static AOT analysis, standalone
     microbenchmarks, or narrower hooks instead.
   - Prefer lower-overhead Level Zero events, SYCL event timing, or tiny
     standalone synchronized diagnostics that do not load the full vLLM model.
   - Measure whether KV writes, attention kernels, or projection kernels now dominate after the u4 MoE bridge.
   - FP8 KV is not a current throughput path for MiniMax TP4. It doubled
     reported KV capacity from roughly `17,216` tokens to `34,496` tokens at
     `max_model_len=2048`, but the p512/n512 serve benchmark completed zero
     requests and died in `sample_tokens` after shared-memory broadcast
     timeouts. Keep it as a future longer-context/debug track only.

4. MoE epilogue fusion
   - Delaying MoE allreduce and output-projection allreduce gave small short-run improvements but did not repeat above the conservative long-run reference.
   - The useful direction is not just moving collectives; it is combining the u4 expert output, output epilogue, allreduce, residual add, and RMSNorm where graph-safe.

5. Scheduler and graph-cache hygiene
   - Isolate experimental `VLLM_CACHE_ROOT` directories.
   - Always record cold versus warm behavior and GPU KV-cache tokens.
   - Treat the `9,408` KV-token artifact as a performance smell, not a valid speed floor.
   - Keep dormant helper/timing branches out of active runtime unless the corresponding env flag is being tested.
   - Forcing oneCCL to assume fabric vertex connectivity with
    `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` is closed as a negative on this
    four-B70 host. It reused the same quality-cleared AOT graph but reduced
    warm p512/n1536 throughput from `38.046755` to `37.210882` output tok/s.
   - DBO is not a current B70 flag path. vLLM rejects `--enable-dbo` without
     DeepEP all-to-all, and upstream documents DBO as requiring DP>1, expert
     parallelism, and DeepEP. Keep DBO as an overlap design reference only.
   - Decode-context parallelism is also closed for this TP4 MiniMax layout:
     `--decode-context-parallel-size 2` fails GQA/MQA validation because the
     model has 8 KV heads and the four-B70 TP4 shape does not satisfy vLLM's
     DCP constraint.
   - EP round-robin does not rescue expert parallelism for batch-1 speed.
     The warm p64/n32 screen reached only `22.338` output-equivalent tok/s,
     consistent with earlier slower EP4 long-run behavior.

6. Speculative decode and MTP
   - Current MiniMax n-gram, ngram_gpu, and DFlash screens are negative or unstable.
   - A fresh p64/n128 screen confirmed the gap: clean baseline reached
     `50.618` total tok/s and `33.745` output tok/s, while plain ngram
     reached only `11.059` total tok/s and `7.373` output tok/s. Raising
     `MAX_BATCHED_TOKENS` to `1024` did not change the result materially.
   - `ngram_gpu` is still not usable on this XPU stack. It loaded and compiled,
     but stalled at zero processed prompts until terminated.
   - A fast-NVMe DFlash retest loaded and compiled both target and drafter, but
     still stalled before generating a p64/n32 sample. Storage was not the
     blocker.
   - Revisit speculation only with a draft that has target-model verification and a measured acceptance rate high enough to beat the extra scheduler and KV pressure.
   - vLLM's native suffix speculative path is now technically runnable on this
     XPU stack via a minimal Arctic `suffix_decoding` subset. A p64/n16 random
     smoke was effectively tied with no-spec: `18.322240` total tok/s with
     suffix versus `18.192808` total tok/s without suffix. Because suffix
     disables async scheduling, this is not a speed result; revisit only with
     repetitive/cache-friendly prompts and acceptance behavior recorded.
   - If speculation works, publish both decode tok/s and total/prefill tok/s to LocalMaxxing with the spec method and draft model recorded.
   - Raise the speculative aspiration above the non-spec target: if target
     verification works and acceptance is high enough, `75+` output tok/s is a
     reasonable stretch. Do not label workload-dependent n-gram wins as a
     general MiniMax speedup unless acceptance behavior and prompt shape are
     reported.

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
- The stock `fuse_allreduce_rms` flag was explicitly disabled by the local XPU
  runtime in a p64/n32 screen. Treat upstream AllReduce+RMS as a design pattern,
  not as an available B70 flag path.
- llm-scaler core ESIMD fused add/RMS/GEMV kernels were built locally but are
  unsafe on this stack: direct calls segfaulted in `libsycl.so.8`, and vLLM
  worker imports failed in SYCL image registration while the generated core
  extension was present. The generated core `.so` is quarantined; the working
  `moe_int4_ops` extension remains active.
- The MiniMax AutoRound model config keeps router/gate layers at 16-bit/float
  precision, so INT4 router fusion is not quality-preserving unless separately
  validated. Keep router quality intact and target expert kernels or collective
  epilogues instead.
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
5. Keep `scripts/summarize-vllm-aot-collectives.sh` in the loop. The current
   clean TP4 AOT census reports `92` allreduce comment lines and `48`
   allreduce-to-RMS/MoE boundary lines; this is now the primary fusion target.
6. Do not enable `--enforce-eager` for MiniMax throughput. It is mentioned in
   Intel AutoRound deployment docs, but local p64/n128 throughput dropped from
   `33.745` output tok/s compiled to `18.007` output tok/s eager.
7. The smaller MoE-router fusion track is now mostly closed for this runtime.
   A MiniMax-specific native `top8 from logits -> u4 tiny MoE` path matched the
   exact sigmoid+bias routing rule in synthetic XPU checks, but warmed p512/n512
   reached only `37.948` output tok/s versus the accepted `39.610585`
   reference. A native candidate-repair op registered inside the working
   `moe_int4_ops` extension and matched exact PyTorch candidate repair
   (`ids_equal=True`, max weight diff about `3e-8`), but top16 warmed to only
   `36.324` output tok/s and top12 to `35.942`. Keep both as patch artifacts,
   not active runtime paths.
8. The old c158 cache lead is closed as non-reproducible with current source.
   vLLM refuses the old binary with `Source code has changed since the last
   compilation` and recompiles current `3b096...`. Treat the old `41.130667`
   p512/n1536 number only as a scheduling clue.
9. The vLLM benchmark `--async-engine` is a small positive on the current
   static decode graph. It reached `45.648` output tok/s at p512/n512, then
   repeated at `47.743` and `48.093` output tok/s on p512/n1536. This is now
   the accepted best, but it is only a roughly one-percent runtime-side gain,
   so the path to `60 tok/s` still requires source-level execution work.
   `--stream-interval 32` and `--max-num-seqs 2` are closed as negative.
10. `use_inductor_graph_partition=true` is the first post-CCL quality-preserving
   software win. It changes the compiled graph to AOT `c3f2...`, keeps Q/K
   variance allreduce visible, and repeated at `39.881` then `40.210` output
   tok/s for p512/n1536. Keep it as the preferred long-run benchmark flag while
   we analyze why p512/n512 remains below the older short-run reference.
11. The llm-scaler core `esimd_resadd_norm_gemv_int4_pert` helper is closed as
   a MiniMax projection-fusion lead. A synthetic TP4 probe found a
   cross-workgroup residual mutation race: the actual `o_proj` shape
   (`N=3072,K=1536`) had about `10.3%` fused relative error against a dequant
   reference. A temporary no-store diagnostic confirmed the race but was slower
   than oneDNN INT4-only on that shape, so this helper should not be wired into
   vLLM as-is. Repro script:
   `benchmarks/b70_resadd_norm_gemv_int4_race_probe.py`.
12. `gpu_memory_utilization=0.95` with the current best async/static graph is a
   capacity setting, not a repeatable speed win. It produced one p512/n1536
   run at `48.42` output tok/s but repeated at only `46.21`. `mode=3` combined
   with graph partition and `compile_sizes=[1]` produced the same AOT hash as
   the current best but hit the `9,408` KV-token cold-cache artifact and only
   reached `33.24` output tok/s at p512/n512.

Current code-direction preference after the latest negative screens:

1. Keep the active runtime on the clean quality path for baseline runs.
2. Avoid Python custom-op wrappers around `dist.all_reduce`; they add clones and
   dispatch overhead without changing the real boundary.
3. Implement the next fusion in C++/SYCL or a real compiler pass, starting with
   the smallest MiniMax-specific shape: TP4, decode-sized tensors, Q/K RMS or
   hidden-state allreduce followed immediately by residual add/RMSNorm.
4. If a fusion cannot beat the current p512/n1536 anchor, archive it as a
   negative and leave the env flag unset.

## 2026-05-12 Static Decode Compile Update

`use_inductor_graph_partition=true` plus `compile_sizes=[1]` is now the best
quality-cleared MiniMax AutoRound TP4 recipe. Warm p512/n512 reached
`45.430028` output tok/s and long p512/n1536 repeated at `47.376673` and
`47.586110` output tok/s (`63.448146` total tok/s on the better repeat).
LocalMaxxing accepted the repeated long run as `cmp2mf1zw007wrm01op7aimhk`.

This preserves the Q/K RMS variance allreduce path and still uses no
speculation, no expert dropping, and stock GPU power limits. It also exposed an
Intel `ocloc` / IGC internal compiler error during the cold static single-token
compile for `triton_red_fused__to_copy_mm_t_9`; preserve that as a driver
compiler bug lead. The next target remains `60+` output tok/s through real
collective/RMS/projection/MoE boundary fusion or a stable speculative path.

Follow-up screens closed several nearby compiler-shape branches. Lowering
`max_num_batched_tokens` to `512`, adding `compile_ranges_endpoints=[512]`,
adding static prefill `compile_sizes=[1,512]`, and disabling combo kernels all
regressed to roughly `30-33` output tok/s or failed during compile/KV setup.
The useful serving tradeoff is `gpu_memory_utilization=0.95` with the winning
decode-only graph: p512/n1536 reached `46.86` output tok/s while raising the
reported KV budget to `33,024` tokens at the default 2k configured window.
Configured 32k still failed after graph memory overhead (`1.52 GiB` available
for KV versus `1.94 GiB` required), while configured 24k succeeded with `25,600`
KV tokens and `33.81` output tok/s at p512/n512; LocalMaxxing accepted the 24k
capacity datapoint as `cmp2p5jhb009wrm01cmkurcfa`. Forced XPU PIECEWISE graph mode
remains closed for speed, even with the local no-op communicator graph-capture
guard: a p64/n32 smoke reached only `4.94` output tok/s.

Additional static-compile retests closed three older source-level helper
branches under the new best recipe. XPU MiniMax Q/K helper fusion plus
`compile_sizes=[1]` reached only `6.39` output tok/s on p64/n32. The exact
MiniMax logits MoE path activated, but p512/n512 reached only `35.06` output
tok/s and forced an extra `(1,1024)` compile graph with `9,408` KV tokens.
Attention delayed-allreduce plus static compile completed a p64/n32 smoke at
only `6.61` output tok/s. Keep all three flags unset and focus remaining
60 tok/s work on a real XPU collective/epilogue fusion or a verified
speculative path.

## DP4+EP Status

DP4+expert-parallel is now partially unblocked. A local XPU worker patch that
sets `CCL_LOCAL_RANK` and `CCL_LOCAL_SIZE` from the DP/TP/PP topology lets
four local DP ranks initialize oneCCL under vLLM serving.

Current status:

- DP4+EP eager no-scaler smoke completed at p16/n8, proving initialization and
  generation.
- DP4+EP eager llm-scaler safe-id smoke also completed, but was slower than the
  no-scaler smoke.
- DP4+EP compiled mode failed even with max autotune disabled and
  `MAX_BATCHED_TOKENS=128`: each rank loaded about `30.81 GiB` of model state,
  leaving only about `677 MiB` free, then Inductor tried to allocate about
  `1.15 GiB`.

Decision: keep the CCL local-rank patch archived as a real reproducibility fix.
Do not promote EP as a speed path yet. The current four-B70 MiniMax speed work
returns to TP4 collective-boundary fusion unless we find a way to reduce EP
runtime memory enough for compiled mode.

## 2026-05-10 Refresh

The AutoRound MiniMax target is now explicitly higher than the old GGUF target.
For the full MiniMax AutoRound model on four B70s, treat `60 tok/s` p512/n1536
decode as the main software target and `75+ tok/s` as a speculation-only stretch
that must still be target-verified. Current useful public results should include
both output tok/s and total tok/s so prefill effects are visible.

External references checked today:

- `intel/llm-scaler-vllm:0.14.0-b8.2.1` is the current public llm-scaler vLLM
  image tag visible on Docker Hub, while the April 2026 B70 release notes and
  reports describe official Arc Pro B70 support in the `0.14.0-b8.2` family.
- vLLM's current fusion docs list sequence parallelism as
  `AllReduce -> RMSNorm` to `ReduceScatter -> local RMSNorm -> AllGather`, but
  the docs frame it as CUDA-tested and not a ready XPU path.
- vLLM's XPU kernel package has moved normalization kernels such as RMSNorm and
  fused add-RMSNorm into an out-of-tree importable kernel package. That is useful
  for reference, but local provider swaps alone have already benchmarked below
  the current MiniMax reference. The missing piece remains collective-boundary
  fusion, not just a faster standalone norm kernel.

## 2026-05-10 Follow-Up Decisions

Native MiniMax MTP is blocked for the current AutoRound checkpoint. The config
advertises `use_mtp=true`, `num_mtp_modules=3`, and `num_hidden_layers=62`, but
the safetensors index has no `model.layers.62+` weights. The local vLLM MTP
speculative mapping also does not include `minimax_m2`. Do not treat the config
flag as usable MTP unless a checkpoint with MTP weights appears or a compatible
drafter is implemented and target-verified.

Suffix decoding remains interesting, especially for repetitive coding or agentic
workloads, but it should be isolated. The active XPU venv does not have
`arctic_inference`, and the current PyPI package is source-only with build
metadata that pins `torch==2.7.0`. Do not install it blindly into
`/home/steve/.venvs/vllm-xpu`; use a separate venv or source checkout first.

DP4+EP compiled mode was retested with lower KV reservation and with Inductor
combo kernels disabled. Both probes failed at the same point: after each rank
loaded about `30.81 GiB` of model state, Inductor attempted a `1.15 GiB` XPU
allocation with only about `0.65-0.68 GiB` free. Keep DP4+EP compiled closed
until per-rank model memory drops or the XPU compile/autotune path avoids that
scratch allocation. The CCL local-rank patch is still worth preserving because
it makes DP4+EP eager initialize and generate, but EP is not a speed path yet.

Current implementation focus stays on TP4:

- Q/K variance allreduce plus RMS apply fusion.
- Hidden-state allreduce plus residual/RMSNorm fusion.
- MoE output allreduce plus downstream epilogue fusion.
- Attention/KV scheduling only where it removes a visible launch, fence, or
  collective boundary.
