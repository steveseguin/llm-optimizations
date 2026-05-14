# MiniMax CUDAGraph Copy-Inputs Screen

Date: 2026-05-13

This screen tested `cudagraph_copy_inputs=true` on the current validated
MiniMax M2.7 AutoRound recipe. The change is quality-neutral in intent: it
only changes graph input staging for the PIECEWISE XPU graph path.

## Setup

Base recipe:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- TP4, FP16 activations
- XPU graph with graph partitioning
- llm-scaler INT4 MoE path
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `--block-size 256`
- `MAX_BATCHED_TOKENS=512`
- `MAX_NUM_SEQS=1`
- `--no-enable-prefix-caching`
- no speculative decode
- no expert dropping
- no power-limit or clock changes

Delta:

```json
{
  "use_inductor_graph_partition": true,
  "compile_sizes": [1],
  "cudagraph_mode": "PIECEWISE",
  "cudagraph_copy_inputs": true
}
```

Cache root:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-copyinputs-20260514T004253Z
```

AOT hash:

```text
65cf16d3072bc078047ab6a81ab4ebcdaed377edd31d6d3c29bbd44935034ce2
```

## Result

| Run | Prompt/output | Total tok/s | Output tok/s | Decision |
| --- | ---: | ---: | ---: | --- |
| warmup / compile | 512/128 | `83.091359` | not comparable | cold compile only |
| measurement | 512/1536 | no result | no result | unstable / reject |

The measurement directly loaded the AOT cache, then failed before producing a
benchmark JSON. The worker errors were:

- `UR_RESULT_ERROR_OUT_OF_RESOURCES`
- `UR_RESULT_ERROR_DEVICE_LOST`

The kernel log also recorded xe resets on all four B70s during the failed
screen:

- `0000:e3:00.0`
- `0000:83:00.0`
- `0000:03:00.0`
- `0000:a3:00.0`

## Decision

Do not promote and do not submit to LocalMaxxing. Keep
`cudagraph_copy_inputs` at the default for the current-best recipe.

This is a driver/compiler stability negative, not a quality or model-quality
finding. It is useful mainly as a guardrail: the current 73 tok/s recipe should
avoid this staging option unless the XPU graph implementation changes.

