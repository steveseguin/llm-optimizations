# MiniMax MoE MiniMax-Logits Callable Cache: Marginal

Date: 2026-05-19

## Summary

Tested a default-off Python-side cleanup for the MiniMax llm-scaler W4A16 MoE
logits work-sharing path:

```bash
VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1
```

The patch caches the selected MiniMax llm-scaler custom-op callable on each
MoE layer during `process_weights_after_loading`, instead of importing and
selecting the callable inside `apply_monolithic` on the decode path.

## Quality

The candidate passed the full strict gate:

- raw145 n64 exact hash
- raw145 n256 exact hash
- semantic suite, 2 repeats
- arithmetic repeat, 16 repeats
- extended sixpack, 2 repeats

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-cache-minimax-op-full-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T070952Z-summary.json
```

## Performance

MiniMax M2.7 AutoRound INT4 W4A16, vLLM/XPU TP4, p512/n1536, 4x B70:

```text
output tok/s: [88.75699968565715, 88.06328138691603, 88.57385080018814, 88.80292985726116]
total tok/s:  [118.34266624754287, 117.41770851588805, 118.09846773358419, 118.40390647634821]
mean output: 88.54926543250562
mean total:  118.06568724334082
```

This is only about `+0.047` output tok/s over the current clean high
(`88.501953` output tok/s), so it is inside normal run variance and should not
be treated as a material optimization.

## Same-Session Baseline Attempt

I attempted a same-session no-cache A/B with
`VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=0`.

The first cold-compile attempt was killed by the shared-memory stall guard
during graph capture after three 60-second shared-memory warnings:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/baseline-compare-no-moe-cache-20260519/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T073742Z.log
```

After warming that cache, a rerun produced:

```text
71.71915210589457 output tok/s / 95.62553614119275 total tok/s
88.16535673554874 output tok/s / 117.55380898073165 total tok/s
third repeat: stalled after AOT load and was guard-terminated
```

Artifacts:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/baseline-compare-no-moe-cache-20260519-rerun/
```

Because this counterfactual did not complete cleanly, I am not using it to
claim an improvement. The clean candidate result itself is valid, but the
observed improvement is too small to promote.

## Decision

Do not submit this as a new LocalMaxxing result. Keep the patch as an optional
experiment if useful for further MoE boundary work, but the current promoted
recipe remains the direct Q/K in-place scale path unless a larger repeatable
delta is shown.

## Reproduction

Full strict run:

```bash
LABEL=minimax-moe-cache-minimax-op-full-20260519 \
VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0 \
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1 \
VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1 \
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0 \
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2 \
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0 \
VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0 \
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=4 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```
