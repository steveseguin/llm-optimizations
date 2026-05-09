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
| decode-only vLLM p1/n8, default IPC | fail/hang | rank 1 spun in `libze_intel_gpu.so.1` / `urEventWait`; killed after stack capture |
| decode-only vLLM p1/n8, pidfd IPC | pass | 18.237 total tok/s, functionality smoke only |
| decode-only vLLM p512/n512, pidfd IPC | pass/negative | 70.472 total tok/s, about 35.236 output tok/s; slower than stable u4 decode baselines |

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

## Decode-Only Follow-Up

I changed the vLLM integration so `is_monolithic` remains false and only
decode-sized batches (`x.shape[0] <= 4`) can call the logits helper through a
new `should_apply_decode_monolithic(...)` hook. This avoids sending prefill
through the experimental monolithic path.

Commands:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS=1 USE_LLM_SCALER_MOE=1 CCL_IPC=pidfd XPU_GRAPH=0 DTYPE=float16 INPUT_LEN=512 OUTPUT_LEN=512 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4 scripts/bench-vllm-minimax-autoround-xpu.sh
```

The decode-only patch fixed the full-run hang when paired with pidfd IPC, but
it did not improve performance:

| variant | total tok/s | output tok/s convention | log |
| --- | ---: | ---: | --- |
| stable FP16 u4 decode baseline | 72.050 | 36.025 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T202757Z.log` |
| stable BF16 u4 decode baseline | 73.215 | 36.608 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T193458Z.log` |
| decode-only logits, pidfd IPC | 70.472 | 35.236 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T214336Z.log` |

A default-IPC p1/n8 run did not progress beyond XCCL/device setup. The hot
worker stack was in:

```text
libze_intel_gpu.so.1 -> ur::level_zero::urEventWait -> sycl::event_impl::wait
```

That makes the default-IPC failure look like a transport/runtime stall rather
than a math mismatch. The pidfd run is the valid comparison, and it says the
router/logits fusion is currently a small regression.

Artifacts:

```text
/home/steve/llm-optimizations-publish/data/minimax-m27-router-logits-decodeonly-20260509.json
/home/steve/llm-optimizations-publish/patches/vllm-minimax-router-logits-decodeonly-20260509.patch
/home/steve/llm-optimizations-publish/patches/llm-scaler-minimax-u4-logits-20260509.patch
```
