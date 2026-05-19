# Current Promoted Results

Date: 2026-05-19

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`, `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`, `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`, clone-safe compiled allreduce custom-op via `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` plus `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`, direct in-place Q/K variance allreduce+scale via `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`, and final MoE output allreduce moved inside the MoE custom-op boundary via `VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `88.927945` output tok/s, `118.570593` total tok/s, mean of four clean long repeats
- Output tok/s repeats: `[88.422654, 89.595083, 88.524703, 89.169339]`
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- Delta: `+0.48%` output tok/s over the previous clean direct Q/K variance high (`88.501953`), `+0.20%` over the previous warning-prone speed headline (`88.748424`), and `+10.33%` over the earlier MoE-WS FlashAttention/PIECEWISE baseline
- LocalMaxxing: `cmpco63q90052nw01ov1zxvwp`

Primary artifacts:

- Current strict clean high: `notes/2026-05-19-minimax-moe-output-allreduce-inside-customop.md`, `data/minimax-m27-moe-output-allreduce-inside-customop-20260519.json`, `data/localmaxxing-minimax-m27-autoround-moe-output-allreduce-inside-customop-p512n1536-20260519.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-moe-output-allreduce-inside-customop-p512n1536-20260519.response.json`, `patches/minimax-moe-output-allreduce-inside-customop-20260519.patch`
- Current clean direct Q/K variance follow-up: `notes/2026-05-19-minimax-qk-direct-inplace-scale.md`, `data/minimax-m27-qk-direct-inplace-scale-20260519.json`, `data/localmaxxing-minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.response.json`, `patches/minimax-qk-direct-inplace-scale-20260519.patch`
- Cleaner Q/K-helper follow-up: `notes/2026-05-19-minimax-qk-helper-tinyfp32-inplace.md`, `data/minimax-m27-qk-helper-tinyfp32-inplace-20260519.json`, `data/localmaxxing-minimax-m27-autoround-qk-helper-tinyfp32-inplace-p512n1536-20260519.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-qk-helper-tinyfp32-inplace-p512n1536-20260519.response.json`
- Cleaner alias-correct tiny-FP32 in-place path: `notes/2026-05-19-minimax-qkvar-inplace-fp32n2.md`, `data/minimax-m27-qkvar-inplace-fp32n2-20260519.json`, `data/localmaxxing-minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.response.json`, `patches/minimax-qkvar-inplace-fp32n2-20260519.patch`
- Previous warning-prone speed headline: `notes/2026-05-18-minimax-qkvar-skipclone-fp32n2-win.md`, `data/minimax-m27-qkvar-skipclone-fp32n2-win-20260518.json`, `data/localmaxxing-minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.response.json`, `patches/minimax-qkvar-skipclone-fp32n2-20260518.patch`
- Previous clone-safe custom-allreduce baseline: `notes/2026-05-18-minimax-clone-safe-custom-allreduce-win.md`, `data/minimax-m27-clone-safe-custom-allreduce-win-20260518.json`, `data/localmaxxing-minimax-m27-autoround-clone-safe-custom-allreduce-p512n1536-20260518.payload.json`, `data/localmaxxing-responses/minimax-m27-autoround-clone-safe-custom-allreduce-p512n1536-20260518.response.json`, `patches/minimax-clone-safe-custom-allreduce-20260518.patch`
- Previous logits-WS wins: `notes/2026-05-18-minimax-logits-ws-no-attn-delay-small-win.md`, `data/minimax-m27-logits-ws-no-attn-delay-small-win-20260518.json`, `notes/2026-05-18-minimax-logits-ws-strict-win.md`, `data/minimax-m27-logits-ws-strict-win-20260518.json`, `patches/minimax-logits-ws-path-20260518.md`

Previous promoted MiniMax baselines:

- Alias-correct tiny-FP32 in-place op: `88.103866` output tok/s, `117.471821` total tok/s, LocalMaxxing `cmpc1dxgv0052pc01s1j9i37l`.
- Q/K helper plus alias-correct tiny-FP32 in-place op: `88.313105` output tok/s, `117.750807` total tok/s, LocalMaxxing `cmpc5xmm6005jpc01k84dxd14`.
- Direct Q/K variance in-place scale: `88.501953` output tok/s, `118.002604` total tok/s, LocalMaxxing `cmpc8cmqm0060pc016g5l5ukh`.
- Warning-prone tiny-FP32 skip-clone headline: `88.748424` output tok/s, `118.331232` total tok/s, LocalMaxxing `cmpbz7lyc004rpc019jburzqv`.
- Clone-safe custom allreduce without tiny-FP32 clone elision: `87.279129` output tok/s, `116.372172` total tok/s, LocalMaxxing `cmpbsqm4l001qpc0199azisgz`.
- No-attention-delay logits-WS baseline without clone-safe compiled allreduce custom-op: `82.404268` output tok/s, `109.872357` total tok/s, LocalMaxxing `cmpbifcx3013bmn01747cxix8`.
- Delayed-attention logits-WS baseline: `81.758267` output tok/s, `109.011023` total tok/s, LocalMaxxing `cmpay7th600bbmn01v6csyaro`.
- Earlier MoE-WS FlashAttention/PIECEWISE baseline: `80.602755` output tok/s, `107.470340` total tok/s, LocalMaxxing `cmpasdq5v007nmn019elaut3s`.

Recent quality-safe rejections and screens:

- MoE output-allreduce plus callable-cache stack: `VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1` on top of the current strict high passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack. Result: `88.912296` output tok/s / `118.549728` total tok/s. Decision: reject and do not submit to LocalMaxxing because it is `0.015649` output tok/s below the promoted mean. Artifacts: `notes/2026-05-19-minimax-moe-output-ar-plus-moe-cache-negative.md`, `data/minimax-m27-moe-output-ar-plus-moe-cache-negative-20260519.json`.
- MiniMax MoE WS skip-redundant-contiguous: `VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS=1` reused already-contiguous hidden-state and router-logit tensors before the llm-scaler MiniMax MoE WS custom op. It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack. Result: `88.885135` output tok/s / `118.513514` total tok/s. Decision: reject and do not submit to LocalMaxxing because it is `0.042809` output tok/s below the promoted mean. Artifacts: `notes/2026-05-19-minimax-moe-ws-skip-redundant-contiguous-negative.md`, `data/minimax-m27-moe-ws-skip-redundant-contiguous-negative-20260519.json`, `patches/minimax-moe-ws-skip-redundant-contiguous-negative-20260519.md`.
- Source-path `gpu_model_runner.py` repair restored the local `/home/steve/src/vllm` execution path after a source-tree repeat failed `raw145-n256-exact`. The repair passed strict source-path validation and measured `87.976305` output tok/s / `117.301741` total tok/s. No LocalMaxxing submission. Artifacts: `notes/2026-05-19-minimax-source-gpumodelrunner-sync-repeat.md`, `data/minimax-m27-source-gpumodelrunner-sync-repeat-20260519.json`, `patches/minimax-source-gpumodelrunner-sync-repeat-20260519.md`.
- `VLLM_MINIMAX_AR_FUSED_RMS_XPU=1` with a c10d group-name fix passed one strict screen, but a fresh speed-screen label failed `raw145-n256-exact` before benchmark repeats with a deterministic but different combined token hash. Reject as a repeatability/quality risk. No LocalMaxxing submission.
- `VLLM_MINIMAX_QK_RMS_APPLY_TP_SCALE=1` folded the `1 / tp_world` variance scale into the XPU Q/K RMS apply helper. Quality passed, but result was `88.359247` output tok/s / `117.812329` total tok/s, slower than the clean direct in-place scale baseline.
- `VLLM_XPU_INC_FAST_2D_APPLY=1` bypassed reshape/view work in `INCXPULinearMethod.apply()` when the W4A16 input was already 2D contiguous. Quality passed, but result was `87.733425` output tok/s / `116.977900` total tok/s. Active source and venv patches were removed.
- `VLLM_MINIMAX_AR_RMS_XPU=1` added an ordered AR+RMS extension op. A standalone 4-rank microcheck was bit-exact for the ordered helper, but the integrated model ran around `10.10` output tok/s and hit the B70 `triton_red_fused__to_copy_mm_t_6` / `ocloc` 245 / IGC floating-point-exception path. Reject.
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=512` extended the Q/K RMS helper into prompt/profile token ranges. Quality passed, but result was `87.974187` output tok/s / `117.298916` total tok/s. Keep helper max tokens at `4`.
- Disabled diagnostics cleanup with a static finite-trace flag passed full quality but was slower at `88.119325` output tok/s / `117.492433` total tok/s. Keep the Dynamo-compatible timing context shape unchanged.
- MiniMax MoE callable-cache patch passed full quality and measured `88.549265` output tok/s / `118.065687` total tok/s, but the delta was inside normal run noise. Same-cache A/B later measured cache-on `88.404703` and cache-off `88.056668`; reject.
- `VLLM_XPU_GREEDY_SKIP_LOGITS_FP32=1` passed `raw145-n64-exact` but failed `raw145-n256-exact` on the current clean baseline. Reject as not quality-preserving under the current compiled graph/runtime recipe.
- `VLLM_MINIMAX_ATTN_POST_REDUCE_RMS_XPU=1` was bit-exact in a direct helper microcheck but failed integrated `raw145-n64-exact` and compiled slowly. Reject.
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

- Use the MoE output-allreduce-inside-custom-op result as the current strict baseline for future code work.
- Keep `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0`; generic thresholds are quality-safe but slower than dtype-specific tiny-FP32 routing.
- Continue targeting true XPU fused-boundary work: hidden allreduce plus residual/RMSNorm, Q/K variance allreduce plus Q/K RMS apply, MoE output plus epilogue, and final lm-head/projection boundaries.
- Preserve vLLM's proven allreduce semantics unless a candidate has an exact repeatability proof across fresh graph/cache captures.
- Keep strict quality gates as promotion blockers; do not promote logits/router/argmax shortcuts unless they pass raw exact hashes, semantic checks, arithmetic repeat, and extended sixpack.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
