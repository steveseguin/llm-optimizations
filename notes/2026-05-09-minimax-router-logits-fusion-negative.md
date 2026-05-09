# 2026-05-09 MiniMax Router Logits Fusion

## Goal

Remove the per-layer vLLM router/top-k glue from MiniMax AutoRound decode by
passing raw router logits into a default-off llm-scaler u4 helper. The helper
computes top-2 plus the renormalized softmax inside the extension, then calls
the existing unsigned-u4 tiny MoE decode path.

The experiment is gated by:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS=1
```

## Result

This is not promoted.

| check | result | notes |
| --- | --- | --- |
| extension import | pass | oneAPI 2025.3 build imports cleanly |
| standalone FP16 | pass | logits path exactly matched explicit top-k u4 path |
| standalone BF16 | pass | logits path exactly matched explicit top-k u4 path |
| vLLM p1/n8 smoke | pass | 12.803 total tok/s, useful only as a functionality smoke |
| vLLM p512/n512 | fail/hang | model loaded, then generation stalled; run was killed after repeated shared-memory wait messages |

The full run stalled after prompt rendering with repeated:

```text
No available shared memory broadcast block found in 60 seconds.
```

Workers remained running, so the issue is likely an interaction with the
compiled TP4 vLLM scheduling/worker path rather than the standalone kernel math.

## Logs

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260509T210440Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260509T210627Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T211318Z.log
```

## Interpretation

The standalone exact-match result proves the fused top-2 math is viable, and
the tiny vLLM smoke proves the call path can execute. The full benchmark hang
means the optimization is not safe enough to use for real runs yet.

Keep `VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS` unset for normal benchmarks. The next
debug step is to rerun a very small output case with vLLM decode timing and a
worker-side stack trace if the full request stalls again.
