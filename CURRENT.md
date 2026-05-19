# Current Promoted Results

Date: 2026-05-19

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`, `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`, `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`, clone-safe compiled allreduce custom-op via `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` plus `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`, and shape-gated clone elision for tiny FP32 allreduces via `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2`.
- Shape: p512/n1536, ctx2048, batch 1
- Result: `88.748424` output tok/s, `118.331232` total tok/s, mean of four clean long repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- Delta: `+1.53%` output tok/s over the previous clone-safe custom-allreduce promoted result and `+7.53%` over the previous no-attention-delay strict promoted result
- LocalMaxxing: `cmpbz7lyc004rpc019jburzqv`

Primary artifacts:

- `notes/2026-05-18-minimax-qkvar-skipclone-fp32n2-win.md`
- `data/minimax-m27-qkvar-skipclone-fp32n2-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qkvar-skipclone-fp32n2-p512n1536-20260518.response.json`
- `patches/minimax-qkvar-skipclone-fp32n2-20260518.patch`

Cleaner reliability follow-up:

- `notes/2026-05-19-minimax-qkvar-inplace-fp32n2.md`
- `data/minimax-m27-qkvar-inplace-fp32n2-20260519.json`
- `data/localmaxxing-minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.response.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.http.txt`
- `patches/minimax-qkvar-inplace-fp32n2-20260519.patch`
- LocalMaxxing: `cmpc1dxgv0052pc01s1j9i37l`
- Result: `88.103866` output tok/s, `117.471821` total tok/s, mean of four repeats
- Decision: cleaner reproduction path because it removes the PyTorch custom-op alias warning; the `88.748424` skip-clone result remains the slightly faster speed headline.

Previous promoted MiniMax baselines:

- Clone-safe custom allreduce without tiny-FP32 clone elision: `87.279129` output tok/s, `116.372172` total tok/s, LocalMaxxing `cmpbsqm4l001qpc0199azisgz`.
- No-attention-delay logits-WS baseline without clone-safe compiled allreduce custom-op: `82.404268` output tok/s, `109.872357` total tok/s, LocalMaxxing `cmpbifcx3013bmn01747cxix8`.
- Delayed-attention logits-WS baseline: `81.758267` output tok/s, `109.011023` total tok/s, LocalMaxxing `cmpay7th600bbmn01v6csyaro`.
- Earlier MoE-WS FlashAttention/PIECEWISE baseline: `80.602755` output tok/s, `107.470340` total tok/s, LocalMaxxing `cmpasdq5v007nmn019elaut3s`.

Current caveat:

- The tiny-FP32 no-clone path is quality-clean on the current stack, but PyTorch emits a custom-op aliasing warning and says this behavior may become an error in a future release.
- The 2026-05-19 alias-correct follow-up uses a mutating no-return custom op for tiny FP32 allreduces. It passed the same strict quality gate, removed the alias warning, and averaged `88.103866` output tok/s / `117.471821` total tok/s across four repeats. It is the cleaner reproduction path.
- Three of four alias-correct benchmark logs printed `Bad address (src/pipe.cpp:367)` during shutdown after request completion and JSON write. Track this as shutdown noise; it did not affect quality or benchmark JSONs.

Recent rejections and screens:

- `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=4096` passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack, but averaged only `86.386284` output tok/s and `115.181713` total tok/s across four repeats. It is `-1.95%` versus the alias-correct tiny-FP32 in-place path and showed repeated `triton_red_fused__to_copy_mm_t_9` / `ocloc` error-code `245` fallback noise. Do not submit to LocalMaxxing or promote. Artifacts: `notes/2026-05-19-minimax-allreduce-inplace-n4096-negative.md`, `data/minimax-m27-allreduce-inplace-n4096-negative-20260519.json`, `patches/minimax-allreduce-inplace-threshold-20260519.patch`.
- Guarded allreduce shape logging showed decode fp32 Q/K variance at `(1, 2)`/numel `2`, prefill/profile fp32 Q/K variance at `(512, 2)`/numel `1024`, and decode fp16 hidden-state allreduce at `(1, 3072)`/numel `3072`. That explains the `n4096` regression: the broad threshold caught the fp16 decode hidden-state collective. `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=2048` avoided that shape and passed the full strict quality gate, but averaged only `87.105717` output tok/s / `116.140956` total tok/s. Do not submit or promote; keep generic in-place threshold unset or `0`. Artifacts: `notes/2026-05-19-minimax-allreduce-shape-log-and-n2048-negative.md`, `data/minimax-m27-allreduce-shape-log-and-n2048-negative-20260519.json`, `patches/minimax-allreduce-shape-logger-and-n2048-screen-20260519.patch`.
- `MAX_BATCHED_TOKENS=768` on top of the clone-safe custom-allreduce recipe passed early gates but failed the extended sixpack with nondeterministic greedy output. Keep MBT512.
- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1` passed full quality but only reached `82.288077` output tok/s, below the promoted custom-op path.
- `VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1` with internal custom-op clone disabled completed AOT graph compilation but hung before producing the first raw145 n64 quality JSON.

Detailed historical candidate screens remain in `notes/` and `data/`. The local lab copy of `CURRENT.md` preserves the longer running chronology.

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Use the alias-correct tiny-FP32 in-place op as the reproduction-safe baseline for future code work, while measuring against the faster skip-clone speed headline.
- Keep `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0` for promoted runs; the n2048/n4096 screens show generic in-place thresholds are quality-safe but slower than dtype-specific tiny-FP32 routing.
- Fuse Q/K variance allreduce with Q/K RMS apply if it preserves the exact restored-weight output hashes.
- Benchmark raw communication overhead for `(1, 2)`, `(1, 3072)`, `(512, 2)`, and `(512, 3072)` to separate CCL latency from graph/compiler overhead.
- Continue targeting true XPU fused-boundary work: hidden allreduce plus residual/RMSNorm, MoE output plus epilogue, and final lm-head/projection boundaries.
- Investigate the benchmark shutdown `Bad address (src/pipe.cpp:367)` noise so future public results have cleaner logs.
- Keep strict quality gates as promotion blockers; do not promote logits/router/argmax shortcuts unless they pass raw exact hashes, semantic checks, arithmetic repeat, and extended sixpack.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
