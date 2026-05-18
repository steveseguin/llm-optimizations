# Current Promoted Results

Date: 2026-05-18

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`, `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`, and `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `82.404268` output tok/s, `109.872357` total tok/s, mean of four clean long repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- Delta: `+0.79%` output tok/s over the previous strict logits-WS promoted result and `+2.24%` over the earlier MoE-WS FlashAttention/PIECEWISE baseline
- LocalMaxxing: `cmpbifcx3013bmn01747cxix8`

Primary artifacts:

- `notes/2026-05-18-minimax-logits-ws-no-attn-delay-small-win.md`
- `data/minimax-m27-logits-ws-no-attn-delay-small-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-logits-ws-no-attn-delay-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-logits-ws-no-attn-delay-p512n1536-20260518.response.json`
- `notes/2026-05-18-minimax-logits-ws-strict-win.md`
- `data/minimax-m27-logits-ws-strict-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-logits-ws-strict-p512n1536-20260518.response.json`
- `patches/minimax-logits-ws-path-20260518.md`

Previous promoted logits-WS baseline:

- Same exact router-logits-to-work-sharing path with delayed attention allreduce enabled: `81.758267` output tok/s and `109.011023` total tok/s, mean of two strict-gated repeats.
- Confirmation repeat: `81.197954` output tok/s, `108.263938` total tok/s; three-run mean was `81.571496` output tok/s.
- LocalMaxxing: `cmpay7th600bbmn01v6csyaro`
- Artifacts: `notes/2026-05-18-minimax-logits-ws-strict-win.md`, `data/minimax-m27-logits-ws-strict-win-20260518.json`

Previous MoE-WS baseline:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1` without logits-to-WS routing: `80.602755` output tok/s and `107.470340` total tok/s.
- LocalMaxxing: `cmpasdq5v007nmn019elaut3s`
- Artifacts: `notes/2026-05-18-minimax-moe-ws-flash-piecewise-strict-win.md`, `data/minimax-m27-moe-ws-flash-piecewise-strict-win-20260518.json`

Recent quality-safe neutral or negative work:

- MBT boundary: MBT768 passed at `80.876005` output tok/s, MBT832 passed but slowed to `77.795833`, MBT896/1024 showed corruption or exact-hash failure. Keep MBT512. Artifacts: `notes/2026-05-18-minimax-mbt-boundary.md`, `data/minimax-m27-mbt-boundary-20260518.json`.
- MoE delay allreduce: quality passed but slowed to `79.481453` output tok/s. Artifacts: `notes/2026-05-18-minimax-moe-delay-negative.md`, `data/minimax-m27-moe-delay-negative-20260518.json`.
- No-clone/final-hidden-clone retie: quality passed at `80.791520` output tok/s, not material. Artifacts: `notes/2026-05-18-minimax-no-clone-clonefinal-retie.md`, `data/minimax-m27-no-clone-clonefinal-retie-20260518.json`.
- Logits-WS no-clone/final-hidden-clone: quality passed but slowed to `81.021124` output tok/s. Artifacts: `notes/2026-05-18-minimax-logits-ws-noclone-clonefinal-negative.md`, `data/minimax-m27-logits-ws-noclone-clonefinal-negative-20260518.json`.
- Decode-boundary timing: final logits measured around `0.86 ms/token`; local lm-head projection looked larger than TP logits gathering. Artifacts: `notes/2026-05-18-minimax-decode-boundary-timing.md`, `data/minimax-m27-decode-boundary-timing-20260518.json`.
- Candidate-router repair: top16 failed exact token hash; top32 passed quality but slowed to `80.008471` output tok/s. Artifacts: `notes/2026-05-18-minimax-candidate-router-top32-negative.md`, `data/minimax-m27-candidate-router-top32-negative-20260518.json`.
- WS internal scratch reuse: failed raw145 n64 exact canary with NUL/control-token corruption. Artifacts: `notes/2026-05-18-minimax-ws-internal-reuse-reject.md`, `data/minimax-m27-ws-internal-reuse-reject-20260518.json`.
- MoE trace and tile sweep: trace confirmed expected WS up/down kernels; `UP_NTILE=4` passed but slowed to `79.236469`, `UP_NTILE=8` stalled, `DOWN_HTILE=8` failed exact hash. Artifacts: `notes/2026-05-18-minimax-moe-trace-and-tile-negative.md`, `data/minimax-m27-moe-trace-and-tile-negative-20260518.json`.
- Greedy sampler fp32 skip: quality passed but tied/slowed at `81.549421` output tok/s. Artifacts: `notes/2026-05-18-minimax-greedy-skip-logits-fp32-negative.md`, `data/minimax-m27-greedy-skip-logits-fp32-negative-20260518.json`, `patches/minimax-greedy-skip-logits-fp32-negative-20260518.md`.
- WS top-k reuse: failed raw145 n64 exact immediately with NUL/control-token corruption; reverted and default canary passed. Artifacts: `notes/2026-05-18-minimax-ws-topk-reuse-reject.md`, `data/minimax-m27-ws-topk-reuse-reject-20260518.json`, `patches/minimax-ws-topk-reuse-rejected-20260518.md`.
- Safe hidden-state selection: quality passed under the fair default XPU FlashAttention v2 backend at `81.914167` output tok/s / `109.218890` total tok/s, only `+0.19%` over promoted and inside run variance. Four-repeat confirmation passed the same strict gate but averaged `81.379492` output tok/s / `108.505990` total tok/s, `-0.46%` versus promoted. Decision: neutral/no speed gain, not promoted and not submitted to LocalMaxxing. Erratum: the earlier `77.314354` output tok/s safe-selector run used the strict runner's older `TRITON_ATTN` default, so it is quality-valid diagnostic data but not a fair comparison against the promoted FlashAttention baseline. Artifacts: `notes/2026-05-18-minimax-safe-sample-hidden-select-negative.md`, `data/minimax-m27-safe-sample-hidden-select-negative-20260518.json`, `notes/2026-05-18-minimax-safe-hidden-repeatability.md`, `data/minimax-m27-safe-hidden-repeatability-20260518.json`, `patches/minimax-safe-sample-hidden-select-negative-20260518.md`.
- Logits-WS local argmax: quality passed under the fair default XPU FlashAttention v2 backend, but slowed to `72.980385` output tok/s and `97.307181` total tok/s versus the promoted `81.758267` / `109.011023` logits-WS baseline. Decision: do not promote and do not submit to LocalMaxxing; the full-vocab logits/sampler path is not the current bottleneck. Artifacts: `notes/2026-05-18-minimax-logits-ws-localargmax-negative.md`, `data/minimax-m27-logits-ws-localargmax-negative-20260518.json`.
- Logits-WS Q/K RMS helper: quality passed under the fair default XPU FlashAttention v2 backend, but slightly slowed to `81.441928` output tok/s and `108.589237` total tok/s versus the promoted `81.758267` / `109.011023` logits-WS baseline. Decision: do not promote and do not submit to LocalMaxxing; replacing only local Q/K RMS math does not remove the decode-critical Q/K variance collective boundary. Artifacts: `notes/2026-05-18-minimax-logits-ws-qk-rms-helper-negative.md`, `data/minimax-m27-logits-ws-qk-rms-helper-negative-20260518.json`.
- Logits-WS Q/K helper in-place allreduce: quality passed on top of the current no-attention-delay baseline, but slowed to `81.939211` output tok/s and `109.252281` total tok/s versus current `82.404268` / `109.872357`. Decision: do not promote and do not submit to LocalMaxxing; in-place Q/K variance allreduce is quality-safe but not faster. Artifacts: `notes/2026-05-18-minimax-qk-helper-inplace-allreduce-negative.md`, `data/minimax-m27-qk-helper-inplace-allreduce-negative-20260518.json`, `patches/minimax-qk-helper-inplace-allreduce-negative-20260518.patch`.
- Logits-WS MoE delay allreduce: quality passed under the fair default XPU FlashAttention v2 backend, but slowed to `79.019501` output tok/s and `105.359335` total tok/s versus the promoted `81.758267` / `109.011023` logits-WS baseline. Decision: do not promote and do not submit to LocalMaxxing; coarse delayed MoE residual handling has now lost both before and after logits-WS. Artifacts: `notes/2026-05-18-minimax-logits-ws-moe-delay-negative.md`, `data/minimax-m27-logits-ws-moe-delay-negative-20260518.json`.
- Logits chunked gather: `VLLM_XPU_LOGITS_CHUNKED_GATHER=32768` passed raw145 n64/n256 exact and the semantic suite, but failed the 16-repeat arithmetic gate with token-level nondeterminism. Decision: reject without benchmarking. Artifacts: `notes/2026-05-18-minimax-logits-chunked-gather-reject.md`, `data/minimax-m27-logits-chunked-gather-reject-20260518.json`.

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Keep the MiniMax logits-to-work-sharing FlashAttention/PIECEWISE recipe with attention delay disabled as the strict baseline.
- Target final logits/lm-head cost, hidden-state collective boundaries, MoE/projection epilogue fusion, and prefill efficiency.
- Do not promote logits/router/argmax shortcuts unless they pass the same strict quality gate.
- Avoid logits chunked-gather variants unless there is a new deterministic implementation; `VLLM_XPU_LOGITS_CHUNKED_GATHER=32768` failed repeatability.
- Keep strict runner backend defaults aligned with promoted recipes so candidate comparisons are fair.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
