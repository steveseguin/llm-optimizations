# Q4_0 Reshape-Through-ADD Fusion

Date: 2026-05-05

## Summary

The Qwen3.6 27B Q4_0 GGUF 3x B70 decode path improved from `44.238455 tok/s` to `44.812806 tok/s` by fusing recurrent-layer `linear_attn_out -> RESHAPE -> ADD` sites into the existing SYCL allreduce+residual-add helper.

This is quality-preserving:

- same Q4_0 GGUF weights;
- same f16 KV cache;
- no speculative decoding;
- no sampling change;
- no GPU power-limit change;
- reshape is view-only, and the fused helper writes the same allreduce sum plus mirrored residual into the ADD output.

## Patch

- source: `/home/steve/src/llama.cpp-q4-b70/ggml/src/ggml-backend-meta.cpp`;
- patch artifact: `patches/llama-cpp-sycl-current-q4-reshapeadd-20260505.patch`;
- patch sha256: `000d6b31da654069ea040c0202d47ff64292116d126bdb29082dde0b4a01210f`.
- focused patch-on-prior-fusion artifact: `patches/llama-cpp-meta-reshapeadd-focused-20260505.patch`;
- focused patch sha256: `2860d6f8bd5a1f568002041f33999065e147d15c6a0f60d984f5c732025d623a`.

Implementation detail:

- `cgraph_config` now records `fused_add_via`;
- immediate `partial -> ADD` fusion still works as before;
- new path detects `partial -> RESHAPE -> ADD` with one use, f32 tensors, equal byte size, mirrored residual/output split state, and tensor size <= 64 KiB;
- compute validation accepts the reshape as the ADD input and passes the original partial, residual, and ADD output to `comm_allreduce_add`;
- fallback computes the skipped reshape before the ADD if the backend fusion fails.

## Trace

Command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_META_ALLREDUCE_STATS=4 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 0 -n 1 -r 1 \
  --no-warmup --poll 50 -o jsonl
```

Trace result:

- fused reshape-through-ADD sites: `48`;
- allreduce paths: `127` `backend+add`, `1` plain `backend`;
- allreduce timing summary: `128` x `20480` bytes, `6.097 ms` total, `47.632 us` average;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-probe-triple213-p0n1-20260505T125150Z.log`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-probe-triple213-p0n1-20260505T125150Z.jsonl`.

## Validation

Command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 3 \
  --poll 50 -o jsonl
```

Result:

- prompt: `135.806175 tok/s`;
- decode: `44.812806 tok/s`;
- total: `67.388784 tok/s`;
- decode samples: `44.8839`, `44.8199`, `44.7346`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-triple213-p512n512-r3-20260505T125442Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-triple213-p512n512-r3-20260505T125442Z.log`;
- LocalMaxxing: `cmosmudwl0004k004hzz6l4u6`.

## Next

- Retest 4x B70 with this graph fusion; the previous 4x cliff may improve if the 48 reshape sites were part of the extra synchronization pressure.
- Inspect the final plain decode allreduce, `attn_output-63 -> GET_ROWS`, for a safe fused path.
- If 4x remains poor, move below graph scheduling into a smaller 20 KiB collective implementation or a row-parallel output epilogue.
