# Current Promoted Results

Date: 2026-05-18

## MiniMax M2.7

Current strict quality-passed speed result:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, llm-scaler INT4 MoE decode with work-sharing enabled by `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- Shape: p512/n1536, ctx2048, batch 1
- Result: `80.602755` output tok/s, `107.470340` total tok/s, mean of two strict-gated repeats
- Quality: raw145 exact n64/n256 hashes, semantic suite, 16-repeat arithmetic, and extended sixpack all passed before benchmarking
- LocalMaxxing: `cmpasdq5v007nmn019elaut3s`

Primary artifacts:

- `notes/2026-05-18-minimax-moe-ws-flash-piecewise-strict-win.md`
- `data/minimax-m27-moe-ws-flash-piecewise-strict-win-20260518.json`
- `data/localmaxxing-minimax-m27-autoround-moe-ws-flash-piecewise-strict-p512n1536-20260518.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-moe-ws-flash-piecewise-strict-p512n1536-20260518.response.json`

Recent MBT boundary follow-up:

- `MAX_BATCHED_TOKENS=768`: strict quality passed, `80.876005` output tok/s and `107.834674` total tok/s mean, only +0.34% output over MBT512.
- `MAX_BATCHED_TOKENS=832`: strict quality passed, but slower at `77.795833` output tok/s and `103.727777` total tok/s mean.
- `MAX_BATCHED_TOKENS=896`: unsafe; raw exact canaries passed but semantic repeat produced NUL/control-token corruption.
- `MAX_BATCHED_TOKENS=1024`: unsafe; raw145 n64 exact canary failed immediately with NUL/control-token corruption.
- Decision: keep MBT512 as the promoted public setting. No LocalMaxxing submission was made for the boundary sweep because no candidate gave a material quality-safe improvement.
- Artifacts: `notes/2026-05-18-minimax-mbt-boundary.md`, `data/minimax-m27-mbt-boundary-20260518.json`

## Qwen3.6 27B

The quality-preserving Qwen targets remain separate from MiniMax AutoRound:

- Q4_0 GGUF TP3 remains the current Qwen decode-speed focus.
- Static FP8 TP4 remains the preferred long-context Qwen layout.
- AutoRound/INT4 results should not be compared as equal-quality replacements for FP8/BF16/GGUF without separate quality validation.

## Next Optimization Targets

- Keep the MiniMax MoE work-sharing FlashAttention/PIECEWISE recipe as the new strict baseline.
- Target hidden-state collective boundaries, MoE/projection epilogue fusion, and prefill efficiency.
- Do not promote logits/router/argmax shortcuts unless they pass the same strict quality gate.
- Keep speculative decode optional and quality-gated; no current promoted MiniMax result uses speculation.
