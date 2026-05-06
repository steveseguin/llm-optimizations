# 2026-05-06 Q4 sync-after reduce.wait experiment

## Summary

`GGML_SYCL_COMM_SYNC_AFTER=2` was added as a narrower post-allreduce fence for the single-kernel custom allreduce path. Mode `1` remains the previous known-good diagnostic behavior: wait all participating streams after the allreduce. Mode `2` only waits the root `reduce` event on the host.

This is quality-cleared on the 3x Qwen3.6-27B Q4_0 GGUF tensor-split path and is a small throughput improvement:

- Previous 3x mode `1`: `45.954130 tok/s` decode.
- New 3x mode `2`: `46.194319 tok/s` decode.
- Correctness: 16-step full-logit deterministic repeat passed.

The same mode did not fix 4x scaling:

- Previous 4x mode `1`: `34.920977 tok/s` decode.
- New 4x mode `2`: `34.929313 tok/s` decode.
- 4x short correctness: 8-step full-logit deterministic repeat passed.

## Code change

In `ggml/src/ggml-sycl/ggml-sycl.cpp`, the single-kernel allreduce post-sync now interprets `GGML_SYCL_COMM_SYNC_AFTER` as:

- `0`: no host wait.
- `1`: `wait_all_streams()`.
- `2`: `reduce.wait()`.

Mode `2` preserves the remote-write completion point that 3x needed, without waiting all peer streams.

## Commands

3x quality/perf environment:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3
GGML_SYCL_DISABLE_DNN=1
GGML_SYCL_Q8_CACHE=1
GGML_SYCL_ASYNC_CPY_TENSOR=0
GGML_SYCL_ASYNC_PEER_COPY=1
GGML_SYCL_COMM_ALLREDUCE=1
GGML_SYCL_COMM_SINGLE_KERNEL=1
GGML_SYCL_COMM_EVENT_BARRIER=1
GGML_SYCL_COMM_SYNC_AFTER=2
GGML_META_FUSE_ALLREDUCE_ADD=1
GGML_SYCL_FUSE_MMVQ2=1
```

3x benchmark:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-syncafter2-reducewait-fusemmvq2-triple213-p512n512-r3-20260506T055402Z.jsonl`
- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-syncafter2-reducewait-fusemmvq2-triple213-p512n512-r3-20260506T055402Z.log`
- Prompt: `118.722431 tok/s`
- Decode: `46.194319 tok/s`

3x correctness:

- Path: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/syncafter2-reducewait-20260506T055151Z`
- Status: pass, two independent 16-step full-logit traces matched.

4x benchmark:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-syncafter2-reducewait-fusemmvq2-quad2130-p512n512-r3-20260506T055829Z.jsonl`
- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-syncafter2-reducewait-fusemmvq2-quad2130-p512n512-r3-20260506T055829Z.log`
- Prompt: `86.547242 tok/s`
- Decode: `34.929313 tok/s`

4x correctness:

- Path: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/syncafter2-reducewait-quad-20260506T055606Z`
- Status: pass, two independent 8-step full-logit traces matched.

## 4x root sweep

Short 4x p512/n256/r2 sweep with `GGML_SYCL_COMM_SYNC_AFTER=2`:

- `level_zero:0,1,2,3`: `34.280733 tok/s`
- `level_zero:1,0,2,3`: `34.745079 tok/s`
- `level_zero:2,1,3,0`: `34.797031 tok/s`
- `level_zero:3,2,1,0`: `35.068186 tok/s`

Root order changes are too small to explain the 4x regression. The likely issue is the current single-root collective design: one GPU reads all partials and writes the result back to all peers, so 4x adds more remote traffic without enough local compute reduction to compensate.

## LocalMaxxing

The accurate context/notes row for the 3x quality-cleared mode `1` result was accepted:

- ID: `cmotnobsj0017qu01icxnv6ek`
- Model: `unsloth/Qwen3.6-27B`
- Quantization: `Q4_0`
- Metrics: `45.95413 tok/s` output, `66.202667 tok/s` total
- Context fields: 512 prompt, 512 output, 512 context length

The earlier minimal row `cmotmnnm6000aqu01uzb9wk12` also exists but lacks context and notes. The full `engineFlags` payload still returns `500 Internal Server Error`; the next LocalMaxxing retry should isolate which `engineFlags` key causes the server error.

## Next work

The 4x path needs a different collective, not just a narrower completion wait. Most promising next experiments:

1. Pairwise/tree allreduce with correctness fences, avoiding one root writing all peers.
2. Local write from copied/staged peer partials, but fixing the repeatability failures seen in the first staged-root attempt.
3. 2x2 layout for FP8 or larger models where each pair does local tensor split and higher-level routing/pipeline work reduces per-token allreduce pressure.
