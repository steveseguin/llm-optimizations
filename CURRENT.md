# Current Promoted Results

Date: 2026-05-18

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1` and `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `81.758267` output tok/s, `109.011023` total tok/s, mean of two strict-gated repeats
- Confirmation repeat: `81.197954` output tok/s, `108.263938` total tok/s; three-run mean is `81.571496` output tok/s
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- LocalMaxxing: `cmpay7th600bbmn01v6csyaro`

Primary artifacts:

- `notes/2026-05-18-minimax-logits-ws-strict-win.md`
- `data/minimax-m27-logits-ws-strict-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.response.json`
- `patches/minimax-logits-ws-path-20260518.md`

Previous promoted baseline:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1` without logits-to-WS routing: `80.602755` output tok/s and `107.470340` total tok/s.
- LocalMaxxing: `cmpasdq5v007nmn019elaut3s`
- Artifacts: `notes/2026-05-18-minimax-moe-ws-flash-piecewise-strict-win.md`, `data/minimax-m27-moe-ws-flash-piecewise-strict-win-20260518.json`

Recent MBT boundary follow-up:

- `MAX_BATCHED_TOKENS=768`: strict quality passed, `80.876005` output tok/s and `107.834674` total tok/s mean, only +0.34% output over MBT512.
- `MAX_BATCHED_TOKENS=832`: strict quality passed, but slower at `77.795833` output tok/s and `103.727777` total tok/s mean.
- `MAX_BATCHED_TOKENS=896`: unsafe; raw exact canaries passed but semantic repeat produced NUL/control-token corruption.
- `MAX_BATCHED_TOKENS=1024`: unsafe; raw145 n64 exact canary failed immediately with NUL/control-token corruption.
- Decision: keep MBT512 as the promoted public setting. No LocalMaxxing submission was made for the boundary sweep because no candidate gave a material quality-safe improvement.
- Artifacts: `notes/2026-05-18-minimax-mbt-boundary.md`, `data/minimax-m27-mbt-boundary-20260518.json`

Recent MoE-delay follow-up:

- `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` with the same MBT512 work-sharing FlashAttention/PIECEWISE recipe passed the full strict quality gate.
- Result: `79.481453` output tok/s and `105.975271` total tok/s mean, slower than the promoted baseline.
- Decision: do not promote and do not submit to LocalMaxxing. Keep effort on decode-critical collective and epilogue work.
- Artifacts: `notes/2026-05-18-minimax-moe-delay-negative.md`, `data/minimax-m27-moe-delay-negative-20260518.json`

Recent no-clone/final-hidden-clone follow-up:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` with the same MBT512 work-sharing FlashAttention/PIECEWISE recipe passed the full strict quality gate.
- Result: `80.791520` output tok/s and `107.722027` total tok/s mean.
- Decision: validated tie, not a material new win. Do not submit to LocalMaxxing because the `+0.23%` output delta over the promoted `80.602755` result is within run variance.
- Artifacts: `notes/2026-05-18-minimax-no-clone-clonefinal-retie.md`, `data/minimax-m27-no-clone-clonefinal-retie-20260518.json`

Recent logits-WS no-clone/final-hidden-clone follow-up:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` was retested on top of the current logits-to-work-sharing baseline.
- Result: strict quality passed, but mean speed was `81.021124` output tok/s and `108.028166` total tok/s, below the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: not promoted and not submitted to LocalMaxxing. This makes further flag reties less interesting than measured timing around residual allreduce and final logits boundaries.
- Artifacts: `notes/2026-05-18-minimax-logits-ws-noclone-clonefinal-negative.md`, `data/minimax-m27-logits-ws-noclone-clonefinal-negative-20260518.json`

Recent decode-boundary timing:

- Synchronized diagnostics found final logits at about `0.86 ms/token`, with local lm-head projection larger than TP logits gathering.
- Eager per-layer labels identified three similar steady decode collectives: Q/K variance allreduce, attention delayed residual allreduce, and MoE expert output allreduce.
- Model-forward timing wrappers were not neutral in compiled graph and were reverted. Active `minimax_m2.py` and `logits_processor.py` hashes match the promoted runtime again.
- Artifacts: `notes/2026-05-18-minimax-decode-boundary-timing.md`, `data/minimax-m27-decode-boundary-timing-20260518.json`

Recent candidate-router repair follow-up:

- Top-16 candidate-router repair failed the first raw145 n64 exact token-hash gate, so it was rejected without benchmarking.
- Top-32 candidate-router repair passed the full strict quality gate, including raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Result: `80.008471` output tok/s and `106.677962` total tok/s mean, slower than the promoted `81.758267` / `109.011023` logits-WS baseline.
- Decision: do not promote and do not submit to LocalMaxxing. The candidate-repair path is quality-preserving at top32 but not faster than the exact router-logits WS path.
- Artifacts: `notes/2026-05-18-minimax-candidate-router-top32-negative.md`, `data/minimax-m27-candidate-router-top32-negative-20260518.json`

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Keep the MiniMax logits-to-work-sharing FlashAttention/PIECEWISE recipe as the new strict baseline.
- Target final logits/lm-head cost, hidden-state collective boundaries, MoE/projection epilogue fusion, and prefill efficiency.
- Do not promote logits/router/argmax shortcuts unless they pass the same strict quality gate.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
