# Qwen3.6 27B FP8 on 4x Arc Pro B70: TP4 vs PP2xTP2

Date: 2026-05-04

Model: `vrfai/Qwen3.6-27B-FP8`

Hardware: 4x Intel Arc Pro B70 32GB on Ubuntu 24.04.4 LTS

Runtime: local vLLM/XPU source checkout with the singleton compressed-tensors attention-scale FA2 patch applied. A second language-only patch was added so `Qwen3_5ForConditionalGeneration` avoids constructing the unused vision tower and skips `visual.*` weights when `--language-model-only` is set.

## Takeaways

- TP4 with patched XPU FlashAttention2 remains the preferred Qwen3.6 27B FP8 topology.
- PP2xTP2 loads and generates with default oneCCL `pidfd` IPC, but is much slower for a single active sequence.
- Both TP4 and PP2xTP2 fit the model's configured 262,144-token context.
- FP8 KV cache increases reported KV capacity but is slower here and carries quality risk, so auto/BF16 KV remains the quality-preserving speed path.

## Results

| Topology | Context | Prompt/Output | Result |
| --- | ---: | ---: | ---: |
| TP4, patched FA2, auto KV | 1,024 | 512/512 | 41.503 tok/s output |
| TP4, patched FA2, auto KV | 1,024 | 512/256 | 39.264 tok/s output |
| PP2xTP2, patched FA2, auto KV | 1,024 | 512/256 | 22.721 tok/s output, LocalMaxxing `cmormmlz0000bky04wpu4oc01` |
| TP4, patched FA2, auto KV | 262,144 | 32/8 smoke | fits, 1,206,355 KV tokens reported |
| PP2xTP2, patched FA2, auto KV | 262,144 | 32/8 smoke | fits, 1,138,148 KV tokens reported |
| TP4, patched FA2, FP8 KV | 1,024 | 512/256 | 28.036 tok/s output |

## Notes

Plain TP2 remains blocked by an XPU memory/OOM path around `lm_head` allocation. The language-only vision skip patch is still useful and correct for text-only operation, but does not by itself make two-card TP2 viable.

For PP2xTP2, do not force `CCL_ZE_IPC_EXCHANGE=sockets`: the model loads but oneCCL fails when pipeline point-to-point traffic starts. Default `pidfd` IPC works.

The PP2xTP2 result should be treated as a capacity fallback for larger models, not a speed target for Qwen3.6 27B. TP4 already fits full configured context and is materially faster. The PP2xTP2 512/256 number was submitted to LocalMaxxing as a diagnostic topology result; the max-context smokes and FP8-KV screen were not submitted as primary leaderboard operating points.

## Key Logs

- PP2xTP2 512/256: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-pp2-tp2-fa2-pidfd-w1-i2-in512-out256-20260504T194506Z.log`
- PP2xTP2 max context smoke: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-pp2-tp2-fa2-pidfd-max262k-smoke-in32-out8-20260504T194807Z.log`
- TP4 max context smoke: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-pidfd-max262k-smoke-in32-out8-20260504T195135Z.log`
- TP4 FP8 KV screen: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-vrfai-tp4-fa2-kvfp8-w1-i2-in512-out256-20260504T195521Z.log`
