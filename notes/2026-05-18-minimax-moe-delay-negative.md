# 2026-05-18 MiniMax M2.7 MoE-delay no-promote

This note records a strict quality-gated test of `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` on top of the current 4x B70 MiniMax M2.7 AutoRound baseline.

The candidate preserved quality, but did not improve speed, so it was not submitted to LocalMaxxing.

## Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Runtime shape: p512/n1536, ctx2048, batch 1
- Promoted baseline: `80.602755` output tok/s, `107.470340` total tok/s
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, llm-scaler INT4 MoE decode work-sharing

## Candidate

Same recipe as the promoted baseline, with:

- `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `MAX_BATCHED_TOKENS=512`
- `COMPILATION_CONFIG_JSON={"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}`

The harness unset the older QK RMS helper experiment variables:

- `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION`
- `VLLM_MINIMAX_QK_NORM_C10D_GROUP_NAME`

## Quality

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact hash: passed
- raw145 n256 exact hash: passed
- semantic suite: passed
- arithmetic repeat: 16/16 exact `42`, deterministic
- extended sixpack: passed

## Benchmark

Two p512/n1536 repeats:

| Repeat | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| 1 | 79.558786 | 106.078381 |
| 2 | 79.404120 | 105.872161 |

Mean:

- Output: `79.481453` tok/s
- Total: `105.975271` tok/s

This is about `-1.39%` output throughput versus the promoted `80.602755` tok/s baseline.

## Decision

Do not promote `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` for the current p512/n1536 decode objective. It is quality-safe in this harness, but slower than the existing promoted baseline.

No LocalMaxxing submission was made because this is not a material improvement.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-delay-ws-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T070932Z-summary.json`
- Data: `data/minimax-m27-moe-delay-negative-20260518.json`

## Next steps

- Keep the promoted baseline unchanged.
- Move effort back to real decode hot-path work: QK/RMS collective reduction, MoE/projection epilogue fusion, and graph/framework callback reduction that does not alter logits.
- Treat MoE allreduce delay as a possible future retest only after the surrounding graph/collective implementation changes.
