# MiniMax Eager XPU Timing Diagnostic

Date: 2026-05-13

I ran a short eager diagnostic to get MiniMax M2.7 region timing on XPU:

```text
VLLM_XPU_DECODE_TIMING=1
VLLM_XPU_DECODE_TIMING_SYNC=1
--enforce-eager
--no-enable-prefix-caching
prompt/output: 512/16
```

This is not an optimized throughput benchmark. It disables the AOT/XPU graph
path and synchronizes around timed regions, so the token rate is intentionally
distorted. The point is attribution.

Logs:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n16-20260513T181411Z.log`
- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n16-20260513T181411Z.json`

## Rank 0 Summary

| Region | Count | Total ms | Avg ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| `minimax.moe.experts_total` | 1116 | `2098.495` | `1.880` | `1192.945` |
| `minimax.attn.qk_norm` | 1116 | `874.901` | `0.784` | `343.051` |
| `all_reduce:(512, 3072):torch.float16` | 250 | `528.522` | `2.114` | `449.971` |
| `minimax.attn.kv_attention` | 1116 | `487.817` | `0.437` | `310.651` |
| `all_reduce:(1, 3072):torch.float16` | 2000 | `340.580` | `0.170` | `13.271` |
| `minimax.attn.delayed_residual_allreduce` | 1116 | `288.031` | `0.258` | `8.186` |
| `minimax.moe.router_linear` | 1116 | `226.199` | `0.203` | `63.260` |
| `minimax.attn.qkv` | 1116 | `186.283` | `0.167` | `80.603` |
| `all_reduce:minimax_qk_var:(1, 2):torch.float32` | 992 | `158.074` | `0.159` | `3.394` |
| `minimax.attn.o_proj` | 1116 | `117.465` | `0.105` | `29.939` |
| `minimax.attn.rope` | 1116 | `63.894` | `0.057` | `1.726` |
| `all_reduce:minimax_qk_var:(512, 2):torch.float32` | 124 | `23.316` | `0.188` | `4.593` |

## Findings

- AOT graph replay bypasses the Python `timed_region` helper, so this eager run
  is only an attribution proxy.
- MoE expert execution is the largest timed region, though the max values
  include prefill and eager overhead.
- Decode has many small collectives. The hidden-state allreduce shaped
  `(1, 3072)` averaged about `0.170 ms`; the tiny Q/K variance allreduce shaped
  `(1, 2)` averaged about `0.159 ms`. The tiny collective pays almost the same
  per-call latency as the larger one.

## Next Targets

- Reduce or fuse MiniMax Q/K RMS variance allreduces.
- Revisit allreduce plus residual/RMS epilogue fusion in an XPU-supported form,
  avoiding the slow Python custom-op path that was already tested.
- Inspect llm-scaler INT4 MoE decode kernels and routing overhead for
  `x.shape <= 4`, since MoE remains the dominant compute region.
- Add lower-level instrumentation that survives AOT graph replay.
