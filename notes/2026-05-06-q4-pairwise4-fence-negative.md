# 2026-05-06 Q4 pairwise4 fence experiment

## Summary

The existing `GGML_SYCL_COMM_PAIRWISE4=1` branch is not correct as-is on the
4x Qwen3.6-27B Q4_0 tensor-split path. A two-run full-logit smoke test diverged
at step 0.

Adding conservative host waits under `GGML_SYCL_COMM_SYNC_AFTER=2` made pairwise4
deterministic but too slow:

- Correctness: pass, 8-step full-logit repeat.
- Decode: `24.754713 tok/s`.
- Baseline single-root 4x mode `2`: `34.929313 tok/s`.

Splitting the waits showed the required fence is after the final/copy fanout, not
after the pair reductions:

- Mode `3` (wait pair reductions only): failed, diverged by step 1.
- Mode `4` (wait final/copy only): passed 4-step smoke test.
- Mode `4` decode: `25.314923 tok/s`.

Conclusion: pairwise4 can be made correct, but the current multi-kernel pairwise
dataflow plus host completion waits is not a performance path.

## Files

Existing pairwise failure:

- `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/pairwise4-existing-20260506T061748Z`

Conservative pairwise mode `2`:

- Correctness: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/pairwise4-syncafter2-20260506T062816Z`
- Benchmark JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pairwise4-syncafter2-fusemmvq2-quad2130-p512n512-r3-20260506T063037Z.jsonl`
- Benchmark log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pairwise4-syncafter2-fusemmvq2-quad2130-p512n512-r3-20260506T063037Z.log`

Split wait sweep:

- Summary: `/home/steve/bench-results/qwen36-q4_0-gguf/pairwise4-splitwait-correctness-20260506T064041Z.txt`
- Mode `3`: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/pairwise4-syncafter3-20260506T064041Z`
- Mode `4`: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/pairwise4-syncafter4-20260506T064041Z`
- Mode `4` benchmark JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pairwise4-syncafter4-fusemmvq2-quad2130-p512n512-r3-20260506T064503Z.jsonl`
- Mode `4` benchmark log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pairwise4-syncafter4-fusemmvq2-quad2130-p512n512-r3-20260506T064503Z.log`

## Next direction

Move away from the pairwise branch for performance. The next 4x candidate is a
striped-root allreduce/fused-add path: split the output vector by index range and
let each GPU root reduce/write a slice instead of making one root perform all
remote reads and writes.
