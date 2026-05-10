# MiniMax M2.7 Negative Screens

Date: 2026-05-10

Target remains `Lasimeri/MiniMax-M2.7-int4-AutoRound` on 4x Intel Arc Pro B70 32GB with vLLM XPU, AutoRound INT4 W4A16, `float16`, and llm-scaler U4 MoE decode. The raised aspiration target is 60 tok/s output for quality-conservative TP4, with higher targets only counted for verified speculative/MTP paths.

Quality-cleared comparison anchors:

- p512/n512 TP4: 39.610585 tok/s output, 79.221171 tok/s total.
- p512/n1536 TP4: 37.552538 tok/s output, 50.070051 tok/s total.

## Screens

| Path | p/n | Output tok/s | Total tok/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| Direct XPU Q/K RMS helper | 512/512 | 28.036348 | 56.072695 | Regression |
| llm-scaler MoE logits decode | 512/512 | 35.899368 | 71.798735 | Regression |
| TP2/PP2 four-card layout | 512/512 | 24.976409 | 49.952818 | Regression |
| Generic FP8 KV cache | 512/512 | 28.103614 | 56.207227 | Regression |
| Explicit `fp8_e5m2` KV cache | 512/512 | n/a | n/a | XPU backend failure |

The direct Q/K helper was the most instructive failure. A microbench made the helper look attractive, but the full compiled graph regressed because the helper broke existing INT4 GEMM/RMS fusion while still leaving the Q/K variance allreduce and wait boundary in place. Do not revisit this path unless the helper is fused back into the existing compiled region or the allreduce/wait is eliminated.

The Q/K boundary microbench remains useful as a local diagnostic. At MiniMax layer sizes on TP4, `qk_full_helper` measured roughly `0.028 ms` for 72-token and 512-token batches versus about `0.176-0.178 ms` for the torch reference. The full-model result shows that isolated kernel wins are not enough; the patch must preserve or improve the generated AOT schedule.

The MoE logits path preserves routing semantics but does not beat the current split router plus U4 decode path. TP2/PP2 reduces TP group size but loses too much single-request utilization to pipeline idle time. Generic `fp8` KV works but is slower at 512 context and carries quality risk; explicit `fp8_e5m2` currently fails in XPU FlashAttention metadata with `Unrecognized FP8 dtype: fp8_e5m2`.

## External Notes

- vLLM's speculative decoding docs still point to model-based methods (EAGLE, MTP, draft model, PARD, MLP) as the best low-QPS latency path. N-gram/suffix methods are expected to be modest and our previous n-gram screens matched that.
- The MiniMax-M2.7 NVFP4 model card reports strong Blackwell performance using TP2/EP2, FP8 KV, and specialized MoE runners. That is useful as a target architecture, but the exact stack is NVIDIA-specific and not directly portable to Arc/XPU.
- Upstream MiniMax recommends vLLM/SGLang for local serving and quality-oriented sampling parameters of `temperature=1.0`, `top_p=0.95`, `top_k=40`.

## Next Work

Focus on true communication and kernel boundaries:

- Implement or adapt an XPU allreduce+RMS/projection fusion that keeps the existing INT4 compiled fusion intact.
- Inspect whether XPU can use a single-kernel allreduce or reduce-scatter/all-gather equivalent for decode-size hidden states.
- Continue mining llm-scaler INT4 MoE kernels for decode-specific improvements instead of routing-logits fusion alone.
- Keep speculative decode on the roadmap, but only count it if target verification preserves output quality.
