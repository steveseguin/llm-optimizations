# MiniMax Call-Site Timing Diagnostic, 2026-05-10

## What Was Tried

I added an opt-in call-site label layer around TP allreduces so eager timing
could distinguish:

- vocab embedding allreduce;
- attention output projection `RowParallelLinear` allreduce;
- MoE final output allreduce;
- Q/K RMS variance allreduce;
- llm-scaler u4 MoE calls.

Patch artifact:

- `patches/vllm-xpu-allreduce-callsite-timing-20260510.patch`

This patch is archived as a diagnostic only. It is not left active in the
runtime because it changed compiled graph behavior even when the timing flag was
disabled.

## Useful Result

The p64/n4 eager timing run with per-call prints produced `5480` timing lines:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-callsite-timing-eager-p64n4-print/vllm-minimax-m27-autoround-tp4-p64n4-20260510T220131Z.log`

For decode tokens, the per-rank call counts are exactly:

| Decode operation | Calls/token/rank | Shape |
| --- | ---: | --- |
| Q/K variance allreduce | `62` | `f32[(1,2)]` |
| attention `o_proj` row-parallel allreduce | `62` | `f16[(1,3072)]` |
| MoE final allreduce | `62` | `f16[(1,3072)]` |
| llm-scaler u4 MoE call | `62` | `M=1` |

That confirms the AOT census: decode has `186` TP collectives per token across
the 62 MiniMax layers before counting the vocab embedding/final path and before
counting the MoE kernels.

## Negative Result

Leaving the call-site label branches in the active runtime caused compiled
benchmarks to fall into the slow low-KV compile artifact:

| Run | Total tok/s | Output tok/s | GPU KV tokens | Outcome |
| --- | ---: | ---: | ---: | --- |
| p64/n128, `max_model_len=2048`, labels present | `25.751753` | `17.17` | `9,408` | negative |
| p64/n128, `max_model_len=512`, labels gated but present in source | `25.205022` | `16.80` | `9,472` | negative |

The logs also showed `Failed to read file <frozen os>` during compile. The
important practical conclusion is that Python-side call-site label branches are
not safe to keep in the compiled MiniMax path.

## Recovery

The active runtime was reverted to the original allreduce call sites while
keeping the older opt-in allreduce/MoE timing hook available.

Recovery validation:

| Shape | Total tok/s | Output tok/s | GPU KV tokens | Log |
| --- | ---: | ---: | ---: | --- |
| p64/n128, `max_model_len=512` | `48.006311` | `32.00` | `18,240` | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-plain-after-callsite-revert-m512-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T221754Z.log` |

This is close to the earlier p64/n128 short reference (`49.347` to `50.419`
total tok/s, `32.90` to `33.61` output tok/s). It is not a LocalMaxxing result.

## Next Direction

Use the call-site evidence, but do not use Python call-site wrappers in the
compiled path. The first real optimization target remains a graph-safe XPU
fusion around:

1. `o_proj` hidden-state allreduce followed by residual add/RMSNorm;
2. MoE-final hidden-state allreduce followed by next-layer residual add/RMSNorm;
3. Q/K variance allreduce only after the hidden-state boundary is addressed.

The diagnostic makes the fusion order clearer: two `f16[(1,3072)]` hidden
collective boundaries per layer are the largest repeated communication surface.
