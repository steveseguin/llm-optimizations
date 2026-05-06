# 2026-05-06 Qwen3.6 27B Q4_0 B70 sync-after validation

## Summary

The Qwen3.6-27B Q4_0 GGUF path now has a quality-cleared fast configuration on Arc Pro B70:

- 1x reference from prior fused-MMVQ2 run: 24.60 tok/s decode.
- 2x B70 tensor split: 40.93 tok/s decode, full-logit repeat stable for 16 greedy decode steps.
- 3x B70 tensor split: 45.95 tok/s decode, full-logit repeat stable for 16 greedy decode steps.
- 4x B70 tensor split: 34.92 tok/s decode, short full-logit repeat stable but slower than 3x.

The main correctness issue was not the fused MMVQ2 kernel. It was ordering/visibility around multi-device tensor split:

- `GGML_SYCL_ASYNC_CPY_TENSOR=1` is nondeterministic on 2x tensor split even when custom allreduce is disabled.
- Generic custom allreduce is nondeterministic because it updates tensors in place while peer queues can still be copying those tensors.
- 2x single-kernel custom allreduce is stable when scheduler async tensor copy is off.
- 3x and 4x single-kernel custom allreduce require an explicit post-collective stream completion point: `GGML_SYCL_COMM_SYNC_AFTER=1`.

## Quality-cleared 3x command

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=0 \
GGML_SYCL_ASYNC_PEER_COPY=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_SYCL_COMM_SYNC_AFTER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_SYCL_FUSE_MMVQ2=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 3 --poll 50 -o jsonl
```

Result:

- Prompt: 118.36 tok/s.
- Decode: 45.954 tok/s.
- Total: 66.203 tok/s.
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-triple213-p512n512-r3-20260506T051928Z.jsonl`
- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-triple213-p512n512-r3-20260506T051928Z.log`

Correctness check:

- Full logits hashed for 16 greedy decode steps.
- Two independent runs produced identical JSONL.
- Path: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/syncafter-long-and-4x-20260506T051503Z`

## Other results

2x quality-cleared fast path:

- Decode: 40.934 tok/s.
- Total: 65.069 tok/s.
- Full-logit 16-token repeat passed.
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-2x-singlekernel-no-cpytensor-fusemmvq2-p512n512-r3-20260506T043717Z.jsonl`
- Correctness: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/2x-quality-cleared-longcheck-20260506T043917Z`

4x quality-cleared short-smoke path:

- Decode: 34.921 tok/s.
- Total: 49.701 tok/s.
- Short full-logit repeat passed, but 4x remains slower than 3x.
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-quad2130-p512n512-r3-20260506T051928Z.jsonl`

Stable fallback without custom allreduce:

- 2x: 27.28 tok/s.
- 3x: 22.38 tok/s.
- 4x: 17.03 tok/s.
- This path is repeatable but scales poorly.

## Negative findings

- `GGML_SYCL_ASYNC_CPY_TENSOR=1` is unsafe for tensor split on this stack. It caused divergent full-logit hashes on 2x even with custom allreduce disabled.
- `GGML_SYCL_ASYNC_PEER_COPY=1` was stable in isolation, but it did not materially improve the fallback path.
- `GGML_SYCL_COMM_STAGED_ROOT_COPY=1` is experimental and failed repeatability on 2x/3x/4x. Do not use it as a performance path yet.
- Generic custom allreduce (`GGML_SYCL_COMM_ALLREDUCE=1`, `GGML_SYCL_COMM_SINGLE_KERNEL=0`) remains unsafe and should be fixed or gated.

## LocalMaxxing status

Tried to submit the 3x 45.954 tok/s quality-cleared result to LocalMaxxing, but the API returned `502 Bad Gateway` for both POST and public leaderboard GET. Payload should be retried when the service is healthy.
