# 2026-05-06 Q4_0 allreduce + GET_ROWS fusion test

## Goal

Test whether the final remaining plain decode collective, `attn_output-63 -> GET_ROWS`, can be fused safely and whether that improves Qwen3.6 27B Q4_0 GGUF tensor-parallel decode on 3x B70.

## Patch

Added an off-by-default meta-backend/SYCL helper:

- env gate: `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1`;
- backend proc: `ggml_backend_comm_allreduce_get_rows_tensor`;
- path label: `backend+getrows`;
- exported by the SYCL backend only;
- correctness guard: f32 partials, i32 row indices, f32 output, mirrored `GET_ROWS` result, mirrored index tensor, `src_nbytes <= 64 KiB`, and output bytes no larger than source bytes;
- fallback: if the helper declines, run the normal allreduce and then compute the original `GET_ROWS` node.

Patch artifact:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-getrows-fusion-current-20260506.patch`
- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-getrows-fusion-current-20260506.patch.gz.b64`

## Trace result

Probe command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1 \
GGML_META_ALLREDUCE_STATS=4 \
llama-bench -m Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 0 -n 1 -r 1
```

The intended graph site was captured:

- `partial_probe`: `attn_output-63`, `MUL_MAT`, `nbytes=20480`, next op `GET_ROWS`, `next_nbytes=20480`;
- paths in the trace: `254` x `backend+add`, `2` x `backend+getrows`;
- no plain `backend` allreduce path remained for the traced decode graph.

Artifacts:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-probe-triple213-p0n1-20260506T000738Z.log`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-probe-triple213-p0n1-20260506T000738Z.jsonl`

## Timing results, early stack

Short screen, gate on:

- `p512/n128/r2`;
- prompt: `135.504744 tok/s`;
- decode: `45.129276 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-triple213-p512n128-r2-20260506T000850Z.jsonl`.

Full validation, gate on:

- `p512/n512/r3`;
- prompt: `135.626208 tok/s`;
- decode: `45.375471 tok/s`;
- computed total: `68.000508 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-fuse-triple213-p512n512-r3-20260506T001019Z.jsonl`.

Same-build control, gate off:

- `p512/n512/r3`;
- prompt: `135.671512 tok/s`;
- decode: `45.340867 tok/s`;
- computed total: `67.967329 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-getrows-control-gateoff-triple213-p512n512-r3-20260506T001228Z.jsonl`.

Reference high-water mark before this patch:

- post-reboot reshape-through-ADD 3x control: `45.624065 tok/s` decode, computed total `68.259384 tok/s`.

## Current-stack recheck

After the later Q8 cache, fused MMVQ2, fused MMVQ2+SwiGLU, RMS_NORM+MUL, event-barrier, and sync-after-2 stack was in place, the same GET_ROWS hook became measurable.

Configuration:

```bash
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=0 \
GGML_SYCL_ASYNC_PEER_COPY=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_SYCL_COMM_SYNC_AFTER=2 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_SYCL_FUSE_MMVQ2=1 \
GGML_SYCL_FUSE_MMVQ2_SWIGLU=1 \
GGML_SYCL_FUSE_RMS_NORM_MUL=1 \
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1 \
llama-bench -m Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL2/SYCL1/SYCL3 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 128 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 5 --poll 50
```

Five-repeat full validation, gate on:

- prompt: `197.252755 tok/s`;
- decode: `49.403656 tok/s`;
- computed total submitted to LocalMaxxing: `79.016858 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/getrows-on-tp3-rmsmul-p512n512-r5-20260506T214555Z.jsonl`.

Five-repeat same-build control, gate off:

- prompt: `196.860926 tok/s`;
- decode: `48.827917 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/getrows-off-tp3-rmsmul-p512n512-r5-20260506T214819Z.jsonl`.

Short decode-only A/B:

- off, `p0/n512/r3`: `48.628094 tok/s`;
- on, `p0/n512/r3`: `49.043584 tok/s`.

Correctness:

- `llama-completion`, `-no-cnv`, greedy decode, same prompt and seed;
- baseline and GET_ROWS stdout SHA256 both `2039492ece1be609e945c074396527ae6e0bcaddd2cf82cce6fd847355711214`;
- stdout was identical (`333333333333333`) for this short deterministic probe;
- this is quality-preserving for the tested path: same Q4_0 weights, same f16 KV, no speculative decoding, no sampling change, no power changes.

LocalMaxxing:

- ID: `cmoultsa900h0ld011f0r2hcs`;
- accepted on 2026-05-06 with `49.403656 tok/s` output.

## Interpretation

The early isolated test was neutral, but after the later fused decode stack landed, GET_ROWS became a small reproducible win: about `+0.576 tok/s` over the same-build five-repeat gate-off control and a new Q4_0 GGUF high-water mark at `49.403656 tok/s`.

Keep `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` in the current best TP3 Q4_0 recipe. It is still a narrow final-logits optimization, not the broader lower-level projection epilogue we need for a large jump. Next Q4_0 work should still target fused row-parallel projection plus allreduce/residual epilogues or same-activation multi-GEMV groups.

The result is leaderboard-worthy because it was validated against a same-build control and does not change quality-affecting settings.
