# 2026-05-05 MiniMax M2.7 Q8 Attention Matvec Blocker

## Scope

- Model: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Engine: patched `llama.cpp` SYCL/Level Zero build in `/home/steve/src/llama.cpp-q4-b70`
- Hardware: 4x Intel Arc Pro B70 32GB
- Goal: get MiniMax M2.7 UD-IQ4_XS to execute across the four B70s after the Qwen 27B work showed usable 3x/4x scaling paths.

## Result

MiniMax still does not reach token generation on the SYCL path. The current blocker is earlier than the MoE split helper: the first dense attention projection in block 0.

The failing operation is:

```text
dst='Qcur-0':type=f32;ne=[6144,1,1,1]
src0='blk.0.attn_q.weight':type=q8_0;ne=[3072,6144,1,1]
src1='attn_norm-0':type=f32;ne=[3072,1,1,1]
```

Default reordered MMVQ behavior:

- `RMS_NORM` and the following elementwise `MUL` complete.
- `src1` is quantized to `Q8_1`.
- Execution then hangs or later reports `UR_RESULT_ERROR_DEVICE_LOST`.

Forced DMMV behavior:

- `RMS_NORM` and elementwise `MUL` complete.
- `src1` is converted to fp16.
- The process segfaults immediately after that conversion.

## Evidence

Scheduler trace:

- log: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe60-split-schedcompute-p0n1-20260505T061326Z.log`
- graph reservation reached `3975` nodes and `122` splits.
- split 0 CPU `GET_ROWS` completed.
- split 1 on `SYCL0` began with `44` nodes, first `norm-0/RMS_NORM`, last `ffn_norm-0 (reshaped)/RESHAPE`.
- all five input copies completed.
- the run hung at `split=1 graph_compute_begin backend=SYCL0 n_nodes=44`.

Op-stat run:

- log: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe60-split-opstats-nov-p0n1-20260505T062119Z.log`
- `norm-0/RMS_NORM` completed in `8679.985 us`.
- process aborted with Level Zero `UR_RESULT_ERROR_DEVICE_LOST`.

SYCL debug run:

- log: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe60-split-sycldebug-p0n1-20260505T062638Z.log`
- `ggml_sycl_rms_norm` completed.
- `ggml_sycl_mul` completed.
- `ggml_sycl_mul_mat` began for `blk.0.attn_q.weight` `q8_0`.
- `quantize_row_q8_1_sycl` completed.
- no `q8_0` matvec completion line was printed before timeout.

Forced DMMV run:

- log: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe60-split-dmmv-p0n1-20260505T063128Z.log`
- same `q8_0` attention Q projection reached.
- `to_fp16_sycl` completed.
- process exited with segmentation fault.

## Interpretation

This is not yet a MiniMax MoE expert placement problem. The existing `GGML_SYCL_MUL_MAT_ID_SPLIT*` debugging did not fire because execution never reached the first `MUL_MAT_ID` path.

The next useful MiniMax work should isolate the `q8_0 x vector` SYCL kernel:

1. Add a small targeted repro around a `q8_0` `[3072,6144] x f32 [3072,1]` matvec on one B70.
2. Trace or guard the reordered `reorder_mul_mat_vec_q8_0_q8_1_sycl` path and the DMMV `q8_0` path separately.
3. Add an env-gated CPU or non-reordered fallback for `q8_0` attention projections only, as a smoke test to reach the MoE/expert path.
4. Check whether this MiniMax GGUF quantization mix is unusually hostile to SYCL. A variant with non-`q8_0` dense attention weights may be a faster route to a working four-card large-model run.

## Artifacts

- structured result: `/home/steve/llm-optimization-artifacts/data/minimax-m27-q8-attn-matvec-blocker-20260505.json`
- scheduler trace patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-scheduler-compute-trace-20260505.patch`
- compressed scheduler trace patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-scheduler-compute-trace-20260505.patch.gz.b64`

## LocalMaxxing

Not submitted. No valid throughput metric was produced; this is a debugging blocker rather than a benchmark.
