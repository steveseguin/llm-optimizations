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
