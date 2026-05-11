# MiniMax Call-Site Timing, 2026-05-11

## Purpose

Add default-off labels to the existing XPU timing hook so an eager diagnostic run
can split TP allreduces into MiniMax call sites:

- `minimax_qk_var`: Q/K RMS variance allreduce;
- `minimax_attn_o_proj`: attention output projection allreduce;
- `minimax_moe_output`: MoE output allreduce.

This is not a performance path by itself. It is instrumentation for choosing the
next fusion target.

## Run

Command shape:

```bash
VLLM_XPU_DECODE_TIMING=1
VLLM_XPU_DECODE_TIMING_SYNC=1
VLLM_XPU_DECODE_TIMING_RANK=0
EXTRA_ARGS=--enforce-eager
TP=4 INPUT_LEN=64 OUTPUT_LEN=8 MAX_MODEL_LEN=512 MAX_BATCHED_TOKENS=256
USE_LLM_SCALER_MOE=1 CCL_IPC=default XPU_GRAPH=0 DTYPE=float16
scripts/bench-vllm-minimax-autoround-xpu.sh
```

Log:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-callsite-timing-20260511/vllm-minimax-m27-autoround-tp4-p64n8-20260511T023913Z.log`

JSON:

`/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-callsite-timing-20260511/vllm-minimax-m27-autoround-tp4-p64n8-20260511T023913Z.json`

## Rank-0 Decode Timing

For the 8 generated tokens, each decoder site appears `62 * 8 = 496` times.
Synchronized eager timing on rank 0:

| Label | Count | Total ms | Avg ms/call | Max ms |
| --- | ---: | ---: | ---: | ---: |
| Q/K variance allreduce, `(1,2) f32` | 496 | `55.764909` | `0.112429` | `2.472629` |
| MoE output allreduce, `(1,3072) f16` | 496 | `54.068999` | `0.109010` | `2.632890` |
| Attention output allreduce, `(1,3072) f16` | 496 | `51.976626` | `0.104792` | `0.941847` |
| llm-scaler u4 MoE kernel, `M1` | 496 | `322.949079` | `0.651107` | `270.533416` |

The MoE row has a first-call outlier. Removing the max event gives about
`0.1059 ms/call`, similar to one allreduce call. That matches the recurring
observation that the expert matvec is not the only issue anymore.

Per decode token, the three visible collective classes are approximately:

- Q/K variance: `6.97 ms/token`;
- MoE output: `6.76 ms/token`;
- attention output: `6.50 ms/token`;
- combined visible collectives: about `20.2 ms/token`.

## Interpretation

This confirms the current MiniMax TP4 path is dominated by collective boundaries
and graph scheduling around them. The next quality-preserving speed path should
try to eliminate or fuse collective boundaries:

1. Q/K variance allreduce is tiny but repeated every layer and every token.
   A useful fix needs to fold the collective into a larger Q/K RMS or attention
   preparation kernel; the earlier standalone IPC microkernel was far too slow
   because it did only peer reads for a tiny payload.
2. Attention output and MoE output each perform one hidden-size allreduce per
   layer per token. Replacing post-wait math alone was not enough; the AR+RMS
   SYCL screen stayed below baseline. A real win needs an XPU communication
   primitive or compiler lowering that avoids an opaque op boundary.
3. Optimizing only the llm-scaler INT4 expert matvec is unlikely to deliver the
   requested 60+ output tok/s without also addressing the collectives.

No LocalMaxxing submission: this was an eager synchronized diagnostic run, not a
throughput result.

Structured data:

`data/minimax-m27-callsite-timing-20260511.json`
