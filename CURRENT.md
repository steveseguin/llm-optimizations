# Current Promoted Results

Date: 2026-05-19

## MiniMax M2.7

Current speed headline:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode, clone-safe compiled allreduce custom-op, and shape-gated clone elision for tiny FP32 Q/K variance allreduces.
- Shape: p512/n1536, ctx2048, batch 1
- Result: `88.748424` output tok/s, `118.331232` total tok/s, mean of four clean long repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- LocalMaxxing: `cmpbz7lyc004rpc019jburzqv`

Current clean-path high:

- Same model/hardware/engine shape.
- Change: `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1` calls `vllm.all_reduce_inplace` directly for decode-sized FP32 Q/K variance tensors and then scales `qk_var` in-place at the MiniMax call site.
- Result: `88.501953` output tok/s, `118.002604` total tok/s, mean of four repeats `[88.739272, 88.302351, 88.821529, 88.144660]`
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- LocalMaxxing: `cmpc8cmqm0060pc016g5l5ukh`
- Use this as the reproduction-safe baseline when the PyTorch custom-op alias warning in the faster speed headline is unacceptable.

Primary artifacts:

- `notes/2026-05-18-minimax-qkvar-skipclone-fp32n2-win.md`
- `data/minimax-m27-qkvar-skipclone-fp32n2-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.response.json`
- `patches/minimax-qkvar-skipclone-fp32n2-20260518.patch`
- `notes/2026-05-19-minimax-qk-direct-inplace-scale.md`
- `data/minimax-m27-qk-direct-inplace-scale-20260519.json`
- `data/localmaxxing-minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.response.json`
- `patches/minimax-qk-direct-inplace-scale-20260519.patch`

Previous promoted MiniMax baselines:

- Alias-correct tiny-FP32 in-place op: `88.103866` output tok/s, `117.471821` total tok/s, LocalMaxxing `cmpc1dxgv0052pc01s1j9i37l`.
- Q/K helper plus alias-correct tiny-FP32 in-place op: `88.313105` output tok/s, `117.750807` total tok/s, LocalMaxxing `cmpc5xmm6005jpc01k84dxd14`.
- Clone-safe custom allreduce without tiny-FP32 clone elision: `87.279129` output tok/s, `116.372172` total tok/s, LocalMaxxing `cmpbsqm4l001qpc0199azisgz`.
- No-attention-delay logits-WS baseline without clone-safe compiled allreduce custom-op: `82.404268` output tok/s, `109.872357` total tok/s, LocalMaxxing `cmpbifcx3013bmn01747cxix8`.
- Delayed-attention logits-WS baseline: `81.758267` output tok/s, `109.011023` total tok/s, LocalMaxxing `cmpay7th600bbmn01v6csyaro`.
- Earlier MoE-WS FlashAttention/PIECEWISE baseline: `80.602755` output tok/s, `107.470340` total tok/s, LocalMaxxing `cmpasdq5v007nmn019elaut3s`.

Recent rejections and screens:

- Source-path `gpu_model_runner.py` repair restored the local `/home/steve/src/vllm` execution path after a source-tree repeat failed `raw145-n256-exact`. The repair restored XPU-aware async copy helpers, removed broad timing wrappers from compiled decode/postprocess paths, and restored the optional sampled-token clone guard. Strict source-path validation then passed raw145 n64/n256 exact hashes, semantic suite, arithmetic repeat, and extended sixpack. Result: `87.976305` output tok/s / `117.301741` total tok/s across two p512/n1536 repeats. This was not submitted to LocalMaxxing because it is below the `88.501953` clean high. Artifacts: `notes/2026-05-19-minimax-source-gpumodelrunner-sync-repeat.md`, `data/minimax-m27-source-gpumodelrunner-sync-repeat-20260519.json`, `patches/minimax-source-gpumodelrunner-sync-repeat-20260519.md`.
- `VLLM_MINIMAX_AR_FUSED_RMS_XPU=1` with a c10d group-name fix passed one strict screen, but a fresh speed-screen label failed `raw145-n256-exact` before benchmark repeats with a deterministic but different combined token hash. Reject as a repeatability/quality risk. No LocalMaxxing submission. Artifacts: `notes/2026-05-19-minimax-ar-fused-rms-c10d-repeatability-negative.md`, `data/minimax-m27-ar-fused-rms-c10d-repeatability-negative-20260519.json`, `patches/minimax-ar-fused-rms-c10d-repeatability-negative-20260519.md`.
- `VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1` folded the `1 / tp_world` variance scale into the XPU Q/K RMS apply helper. Quality passed, but result was `88.359247` output tok/s / `117.812329` total tok/s, slower than the clean direct in-place scale baseline. No LocalMaxxing submission.
- `VLLM_XPU_INC_FAST_2D_APPLY=1` bypassed reshape/view work in `INCXPULinearMethod.apply()` when the W4A16 input was already 2D contiguous. Quality passed raw145 n64/n256, semantic, arithmetic repeat, and extended sixpack, but result was `87.733425` output tok/s / `116.977900` total tok/s across two p512/n1536 repeats. The active source and venv patch were removed after testing. No LocalMaxxing submission. Artifacts: `notes/2026-05-19-minimax-inc-fast-2d-apply-negative.md`, `data/minimax-inc-fast-2d-apply-full-repeat-20260519.json`, `patches/minimax-inc-fast-2d-apply-negative-20260519.md`.
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=512` extended the Q/K RMS helper into prompt/profile token ranges. Quality passed, but result was `87.974187` output tok/s / `117.298916` total tok/s. Keep helper max tokens at `4`.
- Disabled diagnostics cleanup with a static finite-trace flag passed full quality but was slower at `88.119325` output tok/s / `117.492433` total tok/s. Keep the Dynamo-compatible timing context shape unchanged.
- MiniMax MoE callable-cache patch passed full quality and measured `88.549265` output tok/s / `118.065687` total tok/s, but the `+0.047` tok/s delta over the clean high is inside normal run noise. Do not promote.
- Exact-shape XCCL microbench found raw decode-sized allreduces at about `15-17 us`; full-model loss is dominated by framework/compiler/graph boundaries around collectives, not raw CCL latency alone.
- `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=4096` and `=2048` both passed quality but were slower than dtype-specific tiny-FP32 routing. Keep generic in-place threshold unset or `0`.
- `MAX_BATCHED_TOKENS=768` on top of the clone-safe custom-allreduce recipe passed early gates but failed the extended sixpack with nondeterministic greedy output. Keep MBT512.
- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1` passed full quality but only reached `82.288077` output tok/s, below the promoted custom-op path.
- `VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1` with internal custom-op clone disabled completed AOT graph compilation but hung before producing the first raw145 n64 quality JSON.

Detailed historical candidate screens remain in `notes/` and `data/`. The local lab copy of `CURRENT.md` may include a longer running chronology than this concise repo status file.

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Use the clean direct Q/K variance in-place scale path as the reproduction-safe baseline for future code work, while measuring against the faster skip-clone speed headline.
- Keep `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0` for promoted runs; generic thresholds are quality-safe but slower than dtype-specific tiny-FP32 routing.
- Continue targeting true XPU fused-boundary work: hidden allreduce plus residual/RMSNorm, Q/K variance allreduce plus Q/K RMS apply, MoE output plus epilogue, and final lm-head/projection boundaries.
- Preserve vLLM's proven allreduce semantics unless a candidate has an exact repeatability proof across fresh graph/cache captures.
- Keep strict quality gates as promotion blockers; do not promote logits/router/argmax shortcuts unless they pass raw exact hashes, semantic checks, arithmetic repeat, and extended sixpack.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
