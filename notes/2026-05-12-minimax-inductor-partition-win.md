# MiniMax Inductor Graph Partition Win, 2026-05-12

## Summary

`use_inductor_graph_partition=true` is the first repeatable quality-preserving
MiniMax M2.7 AutoRound improvement after the CCL/AOT cleanup.

Accepted LocalMaxxing result:

- id: `cmp2kd5ux006frm013il4qu13`
- status: `APPROVED`
- shape: p512/n1536, context length 2048, batch 1
- output tok/s: `40.2098822280064`
- total tok/s: `53.61317630400853`
- AOT: `c3f2b10098683775b74b9bb91c9a44570f4df792c7a1b0061b5df73b6ef18f20`
- payload: `data/localmaxxing-minimax-m27-autoround-inductor-partition-p512n1536-20260512.payload.json`
- response: `data/localmaxxing-responses/minimax-m27-autoround-inductor-partition-p512n1536-20260512.response.json`

This improves the current quality-cleared long-run baseline from `38.046755`
to `40.209882` output tok/s, about `+5.7%`, without speculation, expert
dropping, quantization changes, or power-limit changes.

## Command Shape

```bash
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
CCL_TOPO_P2P_ACCESS=1 \
VLLM_XPU_ENABLE_XPU_GRAPH=0 \
vllm bench throughput \
  --backend vllm \
  --model /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --tokenizer /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --max-model-len 2048 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1536 \
  --random-range-ratio 0 \
  --num-prompts 1 \
  --disable-log-stats \
  --compilation-config '{"use_inductor_graph_partition":true}'
```

## Results

| Run | Shape | Cache/AOT | KV tokens | Total tok/s | Output tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| graph partition cold | p512/n512 | compiled `c3f2...` | `9,408` | `59.276636` | `29.638318` | cold artifact |
| graph partition warm | p512/n512 | loaded `c3f2...` | `16,832` | `77.196141` | `38.598070` | near miss vs short-run reference |
| graph partition long | p512/n1536 | loaded `c3f2...` | `16,832` | `53.175093` | `39.881320` | beats old long baseline |
| graph partition long repeat | p512/n1536 | loaded `c3f2...` | `16,832` | `53.613176` | `40.209882` | submitted |

Quality checks:

- Generated-cache analyzer sees `f32[s72, 2] -> qk_variance` allreduce sites.
- The top-level graph includes the MiniMax Q/K variance sequence:
  `torch.cat([q_var, k_var])`, `_c10d_functional.all_reduce`, `/ 4`, `chunk`.
- No expert dropping, no speculative decode, no model-quality tradeoff.

## Negative Retries

These were tested before finding the graph-partition win:

| Experiment | Shape | Warm output tok/s | Decision |
| --- | --- | ---: | --- |
| direct Q/K helper, max tokens 1 | p512/n512 | `36.164958` | negative; helper boundary remains too expensive |
| Inductor combo kernels disabled | p512/n512 | `35.445767` | negative; current combo kernels are still net-positive |
| `ir_enable_torch_wrap=false` | p512/n512 | `36.262645` | negative |

The important lesson is that a compiler partitioning change improved the
long-run decode path while local helper kernels and wrapper changes did not.
This supports continuing to work at graph-boundary and scheduling level.

## Next Work

- Compare `c3f2...` against current `3b096...` generated cache to identify
  which allreduce/wait boundaries moved and why p512/n1536 improved.
- Try a controlled p512/n1536 sweep around graph partitioning plus other
  non-semantic compiler knobs only one at a time.
- If the graph-partition cache remains stable, consider making it the default
  MiniMax bench flag in `bench-vllm-minimax-autoround-xpu.sh` behind an env
  variable such as `USE_INDUCTOR_GRAPH_PARTITION=1`.
- Continue the real fusion track for Q/K allreduce plus RMS and hidden-state
  allreduce plus residual/RMS or MoE epilogue.

Structured data:

`data/minimax-m27-inductor-partition-win-20260512.json`
