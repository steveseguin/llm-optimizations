# 2026-05-06 Q4 sync-after focused repro

This is the compact, GitHub-safe repro note for the current Qwen3.6 27B Q4_0 GGUF
Arc Pro B70 optimization state. The full generated patch is kept locally because
large connector writes have truncated base64 patch artifacts.

## Local full patch

- Path: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-fusemmvq2-syncafter-current-20260506.patch`
- Size: `130148` bytes
- SHA256: `eecd45715ba4f623f0ebfd4b26b6ea8ce3f53b56f0b108aaa5d0066a9bd9422c`
- Base64 gzip path: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-fusemmvq2-syncafter-current-20260506.patch.gz.b64`
- Base64 gzip size: `27481` bytes
- Base64 gzip SHA256: `907501c1bc9108bf45a659115ed0a390f371e445674ec5170be2ad20df5fee3f`

Do not trust the previously uploaded `patch.gz.b64` copy in GitHub history; it was
deleted after fetch verification showed truncation/corruption.

## Important source touch points

Repository: `/home/steve/src/llama.cpp-q4-b70`

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - Added env gates:
    - `GGML_SYCL_COMM_SYNC_AFTER`
    - `GGML_SYCL_COMM_SYNC_READY`
    - `GGML_SYCL_COMM_STAGED_ROOT_COPY`
    - `GGML_SYCL_FUSE_MMVQ2`
    - `GGML_SYCL_FUSE_MMVQ2_PROBE`
  - `ggml_backend_sycl_comm_allreduce_tensor`
    - Added `wait_all_streams()` diagnostic completion helper.
    - `GGML_SYCL_COMM_SYNC_READY=1` waits ready barriers before the allreduce.
    - `GGML_SYCL_COMM_SYNC_AFTER=1` waits all participating SYCL streams after
      the single-kernel allreduce. This is required for repeatable 3x/4x tensor
      split on the current B70 stack.
    - `GGML_SYCL_COMM_SYNC_AFTER=2` waits only the root `reduce` event after the
      single-kernel allreduce. This is quality-cleared on 3x and slightly faster
      than mode `1`.
    - `GGML_SYCL_COMM_STAGED_ROOT_COPY=1` copies peer partials to root temp
      buffers before reducing. It failed repeatability and should remain off.
  - `ggml_sycl_try_fuse_mmvq2`
    - Fuses adjacent Q4_0 reordered MMVQ graph nodes sharing the same one-token
      activation, currently covering `ffn_gate + ffn_up` and `Vcur + Kcur`.
- `ggml/src/ggml-sycl/mmvq.cpp`
  - Added `mul_mat_vec_q_reorder_fused2`.
  - Added `reorder_mul_mat_vec_q4_0_q8_1_fused2_sycl`.
  - Added `ggml_sycl_op_mul_mat_vec_q_fused2`.
- `ggml/src/ggml-sycl/mmvq.hpp`
  - Declares `ggml_sycl_op_mul_mat_vec_q_fused2`.

## Quality-cleared launch recipe

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

- Decode: `45.954130 tok/s`
- Prompt: `118.362712 tok/s`
- Total: `66.202667 tok/s`
- Correctness: full-logit repeat stable for 16 greedy decode steps.
- Benchmark JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-triple213-p512n512-r3-20260506T051928Z.jsonl`
- Correctness path: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/syncafter-long-and-4x-20260506T051503Z`

## Correctness conclusions

- `GGML_SYCL_ASYNC_CPY_TENSOR=1` is unsafe for tensor split on this stack. It
  diverged in full-logit hashes on 2x even with custom allreduce disabled.
- `GGML_SYCL_ASYNC_PEER_COPY=1` was stable in isolation and remains enabled.
- Generic custom allreduce is unsafe because it copies from peer tensors while
  peer queues can mutate those tensors in place.
- 2x single-kernel allreduce is stable with scheduler async tensor copy off.
- 3x and 4x single-kernel allreduce require `GGML_SYCL_COMM_SYNC_AFTER=1` for
  repeatability.
- 4x is repeatable in a short smoke test but slower than 3x, so the next work is
  reducing collective/root ordering overhead rather than adding more devices.

## Next patch direction

Replace the diagnostic `wait_all_streams()` implementation behind
`GGML_SYCL_COMM_SYNC_AFTER=1` with a narrower stream/event visibility fence. The
performance target is to keep 3x correctness while recovering some of the sync
overhead and making 4x competitive again.

Update: the first narrower mode is implemented as `GGML_SYCL_COMM_SYNC_AFTER=2`
and uses `reduce.wait()`. It improved 3x Qwen3.6 27B Q4_0 from `45.954130` to
`46.194319 tok/s`, passed a 16-step full-logit repeat, but did not improve 4x
(`34.929313 tok/s`). 4x now needs a non-single-root collective.
