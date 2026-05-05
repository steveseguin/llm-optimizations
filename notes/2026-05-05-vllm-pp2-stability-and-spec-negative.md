# vLLM PP2 x TP2 Stability and N-Gram Negative Result

Date: 2026-05-05

## Summary

After fixing the PP2 `self.drafter` AttributeError, PP2 x TP2 can run the static FP8 model without speculative decoding, but PP2+n-gram remains unsafe.

## Valid PP2 Non-Spec Result

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- engine: patched vLLM/XPU `0.20.1`;
- topology: PP2 x TP2 across 4x B70;
- quantization: compressed-tensors FP8;
- prompt/output: 512/128;
- measured iterations: 2 after 1 warmup;
- `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=1024`;
- `CCL_ATL_TRANSPORT=ofi`, default IPC/topology recognition.

Result:

- avg latency: `4.723338066491124 s`;
- output throughput: `27.099479 tok/s`;
- total throughput: `135.497394 tok/s`;
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260505T044235Z.json`;
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260505T044235Z.log`.

This is slower than the validated TP4 FP8+n-gram path and was not submitted to LocalMaxxing.

## PP2 N-Gram Failures

Configuration:

- same model and topology;
- prompt/output: 512/128;
- `GPU_MEM_UTIL=0.80`.

Failures:

- n-gram `num_speculative_tokens=4`, lookup min/max `2/4`:
  - `num_scheduled_tokens=-3`, `total_num_scheduled_tokens=-3`;
  - log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260505T044601Z.log`.
- n-gram `num_speculative_tokens=2`, lookup min/max `2/4`:
  - `num_scheduled_tokens=-1`, `total_num_scheduled_tokens=-1`;
  - log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260505T044829Z.log`.

## Rejected Patch Attempt

I tested changing the running-request scheduler guard in `vllm/v1/core/sched/scheduler.py` from:

```python
if num_new_tokens == 0:
```

to:

```python
if num_new_tokens <= 0:
```

This avoided the negative scheduled-token assert, but then XPU failed with:

```text
vectorized gather kernel index out of bounds
```

That points to stale speculative `-1` placeholders or invalid draft positions still reaching gather/sample code. The guard was reverted in both source and the active venv.

## Decision

- Keep the `getattr(self, "drafter", None)` PP2 metadata patch.
- Quarantine PP2+n-gram until the placeholder/spec-token cleanup path is fixed properly.
- Continue performance work on validated TP4 FP8 and Q4 GGUF paths.
