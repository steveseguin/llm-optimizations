# 2026-05-06 Q4 striped4 collective experiment

## Summary

Tested a 4-card striped-root allreduce/fused-add branch for Qwen3.6 27B
Q4_0 GGUF on 4x Arc Pro B70. The goal was to avoid the single-root bottleneck
by assigning each root stream every fourth output element and writing that
stripe back to all four tensors.

Result: negative for decode speed.

- `GGML_SYCL_COMM_STRIPED4=1`, `GGML_SYCL_COMM_SYNC_AFTER=0`: failed
  repeatability. Two identical 16-token full-logit runs diverged at step 0.
- `GGML_SYCL_COMM_STRIPED4=1`, `GGML_SYCL_COMM_SYNC_AFTER=2`: repeatability
  passed, but decode fell to `21.297448 tok/s`.
- Current best correctness-cleared 4x single-root baseline remains
  `34.929313 tok/s` with `GGML_SYCL_COMM_SINGLE_KERNEL=1` and
  `GGML_SYCL_COMM_SYNC_AFTER=2`.

Conclusion: striped roots do not fix the 4-card scaling issue. The no-wait
variant is unsafe, and the waited variant adds enough ordering overhead to make
it much slower than the single-root allreduce.

## Benchmark

Model:

- `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

Build:

- `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench`

Runtime gates:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3,0
GGML_SYCL_DISABLE_DNN=1
GGML_SYCL_Q8_CACHE=1
GGML_SYCL_ASYNC_CPY_TENSOR=0
GGML_SYCL_ASYNC_PEER_COPY=1
GGML_SYCL_COMM_ALLREDUCE=1
GGML_SYCL_COMM_SINGLE_KERNEL=0
GGML_SYCL_COMM_PAIRWISE4=0
GGML_SYCL_COMM_STRIPED4=1
GGML_SYCL_COMM_EVENT_BARRIER=1
GGML_SYCL_COMM_SYNC_AFTER=2
GGML_META_FUSE_ALLREDUCE_ADD=1
GGML_SYCL_FUSE_MMVQ2=1
```

Command:

```bash
llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 -ngl 99 -sm tensor -ts 1/1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 3 --poll 50 -o jsonl
```

Files:

- Correctness fail, no wait:
  `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/striped4-syncafter0-repeat-20260506T070044Z`
- Correctness pass, waited:
  `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/striped4-syncafter2-repeat-20260506T070314Z`
- Benchmark JSONL:
  `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-striped4-syncafter2-fusemmvq2-quad2130-p512n512-r3-20260506T070550Z.jsonl`
- Benchmark log:
  `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-striped4-syncafter2-fusemmvq2-quad2130-p512n512-r3-20260506T070550Z.log`

Metrics:

- Prompt: `81.674316 tok/s`
- Decode: `21.297448 tok/s`

## Next

Do not pursue the current striped-root branch for speed. Keep it only as a
diagnostic example of the ordering cost. The next useful Q4_0 work is below the
collective layer: reduce per-token Q4/Q8 matvec overhead, remove more graph
launches, or find a 2x2 execution layout that avoids the pathological 4-card
single-session slowdown.
