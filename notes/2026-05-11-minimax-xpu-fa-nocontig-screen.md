# MiniMax XPU FlashAttention No-Contiguous Screen, 2026-05-11

## Goal

Screen a fresh upstream vLLM XPU patch before considering a full MiniMax
p512/n1536 run:

```text
be0dcc29dcfa83659e6857dd73cb527b5986a7f9
[XPU] remove q/k/v force contiguous for flash_attn (#40356)
```

The patch removes forced `q.contiguous()`, `k.contiguous()`, and
`v.contiguous()` before `flash_attn_varlen_func`. This is quality-preserving if
the XPU FlashAttention kernel accepts the input layout, and it could reduce
attention/KV overhead.

## Result

Runtime: MiniMax M2.7 AutoRound INT4 W4A16, vLLM/XPU TP4, FP16 activations,
llm-scaler u4 MoE decode path, XPU graph off, p64/n128, `max_model_len=512`,
`max_num_batched_tokens=256`.

| Run | AOT | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| cold | `ec6e2b...` | 9,472 | `25.518978` | `17.012652` | cold compile artifact |
| warm | `ec6e2b...` | 18,880 | `49.988730` | `33.325820` | neutral/slightly negative |

Reference p64/n128 clean baseline from 2026-05-10:

- total: `50.618235` tok/s
- output-equivalent: `33.745490` tok/s

## Decision

Do not promote for MiniMax on B70. The patch is mechanically compatible, but the
warm smoke is below the clean short-screen reference. I reverted it from the
active runtime after the screen.

No LocalMaxxing submission. Detailed data:
`data/minimax-m27-xpu-fa-nocontig-screen-20260511.json`.

Keep the upstream vLLM `v0.20.2` / latest-main diff on the watch list, but do
not merge the whole tree over the current dirty XPU runtime without a focused
reason.
