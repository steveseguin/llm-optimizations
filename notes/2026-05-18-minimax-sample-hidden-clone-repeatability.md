# MiniMax Sample-Hidden Clone Repeatability Check

Date: 2026-05-18

## Summary

The sample-hidden clone full-logits recipe was re-run three more times after
the strict quality-gated promotion run. The result is stable: mean output
throughput was `66.705727` tok/s, with a tight `66.640348` to `66.813944`
range.

This supports treating the sample-hidden clone recipe as the current strict
quality-safe MiniMax M2.7 AutoRound TP4 baseline on 4x B70. It does not change
the interpretation of the optimization: the speed delta versus the previous
full-logits clone-final baseline is small, so the value is mostly correctness
isolation and a repeatable baseline for deeper model-body work.

LocalMaxxing result already submitted for the promoted two-run quality-gated
measurement: `cmpag0uvt004io101t6rm25o1`.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: vLLM `0.20.1-local`, XPU TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Quantization: AutoRound INT4 W4A16 / INC
- Shape: p512, n1536, ctx2048, batch 1
- Block size: 256
- Prefix cache: disabled
- Temperature: greedy / 0
- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- `VLLM_XPU_CLONE_SAMPLE_HIDDEN=1`
- `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=0`
- `VLLM_XPU_LOCAL_ARGMAX_DECODE` unset
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
- `CCL_TOPO_P2P_ACCESS=1`
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`
- `ZE_AFFINITY_MASK=0,1,2,3`
- Extra args:
  `--async-engine --block-size 256 --no-enable-prefix-caching --attention-backend TRITON_ATTN --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}`

## Results

| repeat | elapsed s | output tok/s | total tok/s |
| --- | ---: | ---: | ---: |
| 1 | 22.989213 | 66.813944 | 89.085259 |
| 2 | 23.041306 | 66.662888 | 88.883850 |
| 3 | 23.049099 | 66.640348 | 88.853797 |
| mean | - | 66.705727 | 88.940969 |

Sample standard deviation for output tok/s: `0.094394`; min/max spread:
`0.260%` of the mean.

## Interpretation

The candidate is repeatable enough to use as the current strict baseline. The
remaining path to >60 tok/s is already solved, but the next target is >70 and
then >75 without lowering quality. The output-tail timing probe showed only
about `0.05 ms/token` in the sample-hidden clone and `0.044 ms/token` in rank-0
output list conversion, so the next meaningful optimization work should target
compiled model-body work: attention, MoE scheduling, and collective boundaries.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/sample-hidden-clone-repeatability-20260518/summary.bench-only.json`
- Benchmark file list:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/sample-hidden-clone-repeatability-20260518/bench-json-files.txt`
- Promoted parent note:
  `notes/2026-05-17-minimax-sample-hidden-clone-fulllogits-promoted.md`
- Promoted parent data:
  `data/minimax-m27-sample-hidden-clone-fulllogits-promoted-20260517.json`
