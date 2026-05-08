# 2026-05-08 MiniMax Direct SYCL And Layer Placement

## Summary

Tested two MiniMax M2.7 follow-ups after the 16.384 tok/s RPC-layer result:

- direct single-process SYCL, to see if we can remove RPC from the valid layer path;
- layer-placement sweeps on the current process-per-GPU RPC path.

Neither produced a new valid speed record.

## Direct SYCL

Direct SYCL with the ik_llama.cpp build fails during `llm_load_tensors`, before benchmark execution. The failure is a regular SYCL backend model-buffer allocation on GPU0, not the split-buffer pool path:

```text
ggml_backend_sycl_buffer_type_alloc_buffer:
sycl::malloc_device returned null: device=0 size=27676983296
max_alloc=34242297856 free=34180378624 total=34242297856
```

An uneven tensor split reduced the failed allocation to 19.028 GB, but Level Zero still returned null:

```text
device=0 size=19027926016 max_alloc=34242297856 free=34180382720 total=34242297856
```

Reducing batch from `2048` to `512` did not change that failed 19.028 GB allocation.

The captured backtrace is:

```text
ggml_backend_sycl_buffer_type_alloc_buffer
alloc_tensor_range
ggml_backend_alloc_ctx_tensors_from_buft
llm_load_tensors
llama_model_load_from_file
```

Interpretation: the current RPC-worker layout is not just overhead. It avoids a single large regular model-buffer allocation that direct single-process MiniMax hits during model load. A future direct-SYCL path likely needs chunked regular-buffer allocation or a way to force large model tensor ranges through an existing split/pool-style allocator.

## RPC Layer Placement Sweep

Current valid baseline command shape:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 1 -ngl 99 -sm layer -ts '<split>' \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

One-repeat placement sweep:

| Label | `-ts` | tok/s |
| --- | --- | ---: |
| even | `1/1/1/1` | 15.625353 |
| light0 | `0.8/1.05/1.05/1.1` | 16.358185 |
| heavy0 | `1.15/1.0/1.0/0.85` | 16.222051 |
| light3 | `1.1/1.05/1.05/0.8` | 16.219277 |
| heavy3 | `0.85/1.0/1.0/1.15` | 16.260938 |

The best placement was `0.8/1.05/1.05/1.1`, but it did not beat the earlier three-repeat best:

```text
16.383602 tok/s, p0/n64/r3, -ts 1/1/1/1
```

Conclusion: layer placement can recover normal run-to-run variation, but it is not the route from 16 tok/s to the 30 tok/s target. The valid layer-mode path has only five scheduler splits, so generic split coalescing is also not expected to help this mode.

## Next

1. Keep the RPC layer baseline as the valid MiniMax capacity/speed reference.
2. Treat direct-SYCL MiniMax as blocked on chunked regular SYCL model-buffer allocation.
3. Focus true speed-up work on quality-correct graph/tensor/expert parallelism, not layer placement.
4. If graph split is revisited, resolve the correctness/performance problem around cross-device reductions. The current branch-fused graph path lowers split count but is not competitive and has reduction-quality uncertainty.
