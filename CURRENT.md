# Current Promoted Results

Date: 2026-05-18

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`, `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`, `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`, and clone-safe compiled allreduce custom-op via `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` plus `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `87.279129` output tok/s, `116.372172` total tok/s, mean of four clean long repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- Delta: `+5.92%` output tok/s over the previous no-attention-delay strict promoted result and `+8.28%` over the earlier MoE-WS FlashAttention/PIECEWISE baseline
- LocalMaxxing: `cmpbsqm4l001qpc0199azisgz`

Primary artifacts:

- `notes/2026-05-18-minimax-clone-safe-custom-allreduce-win.md`
- `data/minimax-m27-clone-safe-custom-allreduce-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-clone-safe-custom-allreduce-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-clone-safe-custom-allreduce-p512n1536-20260518.response.json`
- `patches/minimax-clone-safe-custom-allreduce-20260518.patch`
- `patches/minimax-strict-harness-custom-allreduce-clone-env-capture-20260518.patch`

Previous promoted MiniMax baselines:

- No-attention-delay logits-WS baseline without clone-safe compiled allreduce custom-op: `82.404268` output tok/s, `109.872357` total tok/s, LocalMaxxing `cmpbifcx3013bmn01747cxix8`.
- Delayed-attention logits-WS baseline: `81.758267` output tok/s, `109.011023` total tok/s, LocalMaxxing `cmpay7th600bbmn01v6csyaro`.
- Earlier MoE-WS FlashAttention/PIECEWISE baseline: `80.602755` output tok/s, `107.470340` total tok/s, LocalMaxxing `cmpasdq5v007nmn019elaut3s`.

Recent rejections and screens:

- `MAX_BATCHED_TOKENS=768` was retested on top of the current clone-safe custom-allreduce recipe.
- It passed raw145 n64/n256 exact hashes, semantic suite, and 16-repeat arithmetic, but failed the extended sixpack with nondeterministic greedy token output on the sort/list prompt.
- No benchmark or LocalMaxxing submission was made; keep `MAX_BATCHED_TOKENS=512` for the promoted clone-safe custom-allreduce path.
- Artifacts: `notes/2026-05-18-minimax-clone-custom-allreduce-mbt768-quality-fail.md`, `data/minimax-m27-clone-custom-allreduce-mbt768-quality-fail-20260518.json`

- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1` was tested as a quality-safe alternative to the clone-safe compiled custom allreduce path.
- It passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack with the promoted hashes.
- Result: `82.288077` output tok/s and `109.717436` total tok/s mean, below the promoted `87.279129` / `116.372172` clone-safe custom allreduce result.
- No LocalMaxxing submission was made because this is quality-safe but slower than the promoted baseline.
- Artifacts: `notes/2026-05-18-minimax-functional-outofplace-allreduce-negative.md`, `data/minimax-m27-functional-outofplace-allreduce-negative-20260518.json`

Detailed historical candidate screens remain in `notes/` and `data/`. The local lab copy of `CURRENT.md` preserves the longer running chronology.

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Use the clone-safe custom allreduce MiniMax result as the new strict baseline.
- Target true XPU fused-boundary work: Q/K RMS variance allreduce plus apply, hidden allreduce plus residual/RMSNorm, and attention output allreduce plus post-attention normalization.
- Keep strict quality gates as promotion blockers; do not promote logits/router/argmax shortcuts unless they pass raw exact hashes, semantic checks, arithmetic repeat, and extended sixpack.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
