# MiniMax FlashAttention PIECEWISE Strict Revalidation

Date: 2026-05-18

## Summary

The older May 13 FlashAttention/PIECEWISE MiniMax M2.7 AutoRound TP4 recipe was
retested under the current strict quality harness. It passed every strict gate:
raw145 n64 exact token hash, raw145 n256 exact token hash, semantic suite,
16-repeat arithmetic, and extended sixpack.

Three adjacent p512/n1536 repeats averaged `70.006353` output tok/s and
`93.341804` total tok/s. This is now the strongest current quality-passed
MiniMax result. It is faster than the sample-hidden-clone TRITON path
(`66.705727` output tok/s repeat mean), but below the old `73.306312` May 13
single-run high. Treat this as the honest strict baseline.

No model, quantization, router precision, KV dtype, sampler, speculative
decoding, or power setting was changed.

LocalMaxxing accepted the result as `cmpahyaas002gmn01lk0625he`.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Submitted/base model: `MiniMaxAI/MiniMax-M2.7`
- Engine: vLLM `0.20.1-local`, XPU TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Quantization: AutoRound INT4 W4A16 / INC
- Attention backend: default XPU FlashAttention v2
- Shape: p512, n1536, ctx2048, batch 1
- Block size: 256
- Max batched tokens: 512
- Prefix cache: disabled
- Temperature: greedy / 0
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
- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE` unset
- `VLLM_XPU_CLONE_SAMPLE_HIDDEN` unset
- `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN` unset
- `VLLM_XPU_LOCAL_ARGMAX_DECODE` unset

Benchmark extra args:

```bash
--async-engine \
--block-size 256 \
--no-enable-prefix-caching \
--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
```

The strict harness was patched to allow `ATTENTION_BACKEND=default`, so quality
and benchmark runs can omit `--attention-backend TRITON_ATTN` and use
FlashAttention.

## Quality

Strict quality summary:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This is a strict quality pass, not a speed-only result.

## Results

Strict quality-gated one-run benchmark:

| output tok/s | total tok/s |
| ---: | ---: |
| 69.960334 | 93.280445 |

Three adjacent repeatability runs:

| repeat | elapsed s | output tok/s | total tok/s |
| --- | ---: | ---: | ---: |
| 1 | 22.025850 | 69.736243 | 92.981657 |
| 2 | 21.961097 | 69.941863 | 93.255817 |
| 3 | 21.836497 | 70.340953 | 93.787938 |
| mean | - | 70.006353 | 93.341804 |

Sample standard deviation for output tok/s: `0.307470`; min/max spread:
`0.864%` of the mean.

## AOT And Collectives

Current revalidation used AOT hash:
`03f6a28c070656d44eab4c581bc8dc5295ed123e7c0150c7f596ea24012406b0`.

Classifier summary:

- actual allreduce call lines: `1496`
- actual wait tensor call lines: `1496`
- categories:
  - embedding hidden: `8`
  - Q/K RMS variance: `496`
  - attention output projection hidden: `496`
  - MoE hidden: `496`

The strict revalidation did not reduce the collective count. The gain versus
the TRITON sample-hidden-clone path appears to come from using the faster
FlashAttention recipe while avoiding the newer clone barriers.

## Decision

Promote this as the current honest MiniMax M2.7 AutoRound TP4 baseline on 4x
B70. The old May 13 `73.306312` result remains a useful historical high, but
this `70.006353` mean is the repeatable strict-quality number to compare
against going forward.

## Artifacts

- Strict summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-old-flash-piecewise-cache-revalidate-strict-tp4-ctx2048-mbt512-bs256-20260518T002801Z-summary.json`
- Strict quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-old-flash-piecewise-cache-revalidate-strict-tp4-ctx2048-mbt512-bs256-20260518T002801Z-quality`
- Repeatability summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/old-flash-piecewise-repeatability-20260518/summary.bench-only.json`
- AOT collective classification:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/old-flash-piecewise-repeatability-20260518/aot-collectives-03f6a28c.json`
- Harness patch:
  `patches/minimax-strict-harness-attention-backend-parameter-20260518.patch`

## Next

The next optimization work should target the 187 collectives per generated
decode graph: exact Q/K RMS variance allreduce fusion, hidden allreduce plus
projection/MoE epilogue fusion, and attention boundary cleanup. The output-tail
clone/list path is not the main remaining bottleneck.
