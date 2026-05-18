# MiniMax Q/K Fusion FlashAttention Retest

Date: 2026-05-18

## Summary

`pass_config.fuse_minimax_qk_norm=true` is now quality-safe under the current
default XPU FlashAttention recipe, but it does not improve speed and does not
remove the target Q/K RMS variance allreduce from the generated AOT graph.

Three adjacent p512/n1536 repeats averaged `69.744355` output tok/s and
`92.992473` total tok/s. That is slightly below the promoted strict
FlashAttention baseline mean of `70.006353` output tok/s. Treat this as a
negative optimization result, not a new speed baseline.

LocalMaxxing accepted the result as `cmpajhpgx004qmn01xrhzxluc`.

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
- `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1`
- `CCL_TOPO_P2P_ACCESS=1`

Compile config:

```json
{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}
```

## Quality

The candidate passed the full strict gate:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This matters because earlier Q/K helper screens had failed quality under other
recipes. The current cleaner FlashAttention path is not degrading output.

## Results

Three adjacent repeatability runs:

| repeat | elapsed s | output tok/s | total tok/s |
| --- | ---: | ---: | ---: |
| 1 | 21.943222 | 69.998837 | 93.331783 |
| 2 | 22.060347 | 69.627192 | 92.836256 |
| 3 | 22.066735 | 69.607035 | 92.809380 |
| mean | - | 69.744355 | 92.992473 |

Min/max spread was `0.562%` of the mean. The run is repeatable, but it is not
faster than the promoted strict FlashAttention mean (`70.006353` output tok/s).

## AOT And Collectives

AOT hash:
`7f1422b4de9682e60e5291b8434407b8f49e38907b85537ba584087183bfb1bf`.

Classifier summary:

- actual allreduce call lines: `1496`
- actual wait tensor call lines: `1496`
- categories:
  - embedding hidden: `8`
  - Q/K RMS variance: `496`
  - attention output projection hidden: `496`
  - MoE hidden: `496`

The pass was enabled in vLLM logs, but the generated graph still has the same
187 collectives per decode graph. The next real performance work remains
eliminating or hiding those communication boundaries, not toggling this pass.

## Harness Fix

The strict harness now passes `COMPILATION_CONFIG_JSON` into the quality checks,
not only into the benchmark path. Without that, a candidate could be quality
tested on the default compile config and benchmarked on a different one. This
was the reason the first `qk-helper-fusion-flashdefault-quality` attempt was
invalid as a Q/K-fusion quality check.

Patch artifact. This captures the current strict-harness script delta, including
the `COMPILATION_CONFIG_JSON` quality/benchmark parity fix:
`patches/minimax-strict-harness-compilation-config-json-20260518.patch`

## Artifacts

- Data summary:
  `data/minimax-m27-qk-fusion-flashdefault-no-uplift-20260518.json`
- Strict summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-helper-fusion-flashdefault-quality2-strict-tp4-ctx2048-mbt512-bs256-20260518T011526Z-summary.json`
- Strict quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-helper-fusion-flashdefault-quality2-strict-tp4-ctx2048-mbt512-bs256-20260518T011526Z-quality`
- Repeatability summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/qk-helper-fusion-flashdefault-repeatability-20260518/summary.bench-only.json`
- AOT classification:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/qk-helper-fusion-flashdefault-aot-collectives-20260518.json`
- LocalMaxxing payload:
  `data/localmaxxing-minimax-m27-autoround-qk-fusion-flashdefault-no-uplift-p512n1536-20260518.payload.json`
- LocalMaxxing response:
  `data/localmaxxing-responses/minimax-m27-autoround-qk-fusion-flashdefault-no-uplift-p512n1536-20260518.response.json`

## Next

Move on from this flag as a speed lever. The next optimization target should be
one of:

- hidden allreduce plus attention/MoE epilogue fusion;
- a real Q/K RMS variance collective replacement, verified by the AOT
  classifier dropping the `qk_rms_variance` count;
- runtime variance and memory-pressure reduction so the promoted FlashAttention
  path returns closer to the old 73.3 tok/s single-run high.
