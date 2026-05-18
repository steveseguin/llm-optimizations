# MiniMax Candidate-Router Repair Follow-Up

Date: 2026-05-18

## Goal

Test whether a smaller exact-router repair set can preserve MiniMax M2.7 routing quality while reducing the overhead of the current exact router-logits to llm-scaler work-sharing MoE path.

This branch intentionally disabled `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1` and instead used the candidate-router repair path with `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`.

## Result

Top-16 candidate repair failed the first exact raw token-hash gate:

- Expected raw145 n64 combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed raw145 n64 combined token hash: `a8f3570b8ed4480c708a958eaac3621dd2b473c39415723e1a87c0ce40d73a49`
- Decision: reject without benchmarking.

Top-32 candidate repair passed the full strict gate:

- raw145 n64 exact hash: pass
- raw145 n256 exact hash: pass
- semantic suite: pass
- repeated arithmetic: 16/16 deterministic exact `42`
- extended sixpack: pass

The top-32 benchmark was slower than the current promoted logits-to-work-sharing path:

- Run 1: `80.777087` output tok/s, `107.702783` total tok/s
- Run 2: `79.239855` output tok/s, `105.653141` total tok/s
- Mean: `80.008471` output tok/s, `106.677962` total tok/s
- Delta versus current promoted `81.758267` output tok/s: `-2.140205%`

## Decision

Do not promote and do not submit to LocalMaxxing. This is a useful quality-preserving negative result, but it is slower than the promoted exact MiniMax router-logits WS path and does not materially improve reproducibility or methodology.

The candidate-router repair path is now less attractive than deeper work on the measured decode bottlenecks:

- final logits / lm-head projection cost;
- Q/K variance, attention residual, and MoE output collective boundaries;
- MoE/projection epilogue scheduling and fusion;
- prefill characterization as a separate non-regressing optimization track.

## Reproduction

Top-32 command shape:

```bash
env -u VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS \
  -u VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS \
  -u VLLM_XPU_LOCAL_ARGMAX_DECODE \
  -u VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE \
  LABEL=minimax-candidate-router-top32-ws-strict-20260518 \
  ATTENTION_BACKEND=default \
  RUN_EXTENDED_QUALITY=1 \
  RUN_REPEAT_ARITHMETIC_QUALITY=1 \
  REPEAT_ARITHMETIC_RUNS=16 \
  BENCH_REPEATS=2 \
  QUALITY_TIMEOUT=35m \
  BENCH_TIMEOUT=30m \
  SHM_STALL_MAX_WARNINGS=3 \
  MAX_BATCHED_TOKENS=512 \
  VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
  VLLM_MINIMAX_M2_CANDIDATE_ROUTER_TOPM=32 \
  VLLM_MINIMAX_M2_CANDIDATE_ROUTER_MAX_TOKENS=4 \
  VLLM_MINIMAX_M2_CANDIDATE_ROUTER_XPU_REPAIR=1 \
  VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-candidate-router-top32-ws-strict-20260518 \
  COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}' \
  /home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

Primary artifacts:

- Top-16 summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-candidate-router-top16-ws-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T101243Z-summary.json`
- Top-32 summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-candidate-router-top32-ws-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T101840Z-summary.json`
- Top-32 run 1: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T103412Z.json`
- Top-32 run 2: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T103704Z.json`
