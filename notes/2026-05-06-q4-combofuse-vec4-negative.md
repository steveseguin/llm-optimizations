# Qwen3.6 27B Q4_0 GGUF: graph-consumer fusion and vec4 allreduce checks

Date: 2026-05-06

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

Engine: llama.cpp SYCL AOT BMG-G31 build at `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`

Base runtime gates:

```bash
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

## Combined graph-consumer fusion

Goal: test whether chaining more meta-level consumers after allreduce reduces 4-card overhead enough to fix the quad regression.

Additional gates:

```bash
GGML_META_FUSE_ALLREDUCE_RESHAPE=1
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1
```

Short 4-card check, `-p 512 -n 128 -r 2`, selector `0,1,2,3`:

- fusion off: `33.987384 tok/s` decode
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-combofuse-off-quad0123-p512n128-r2-20260506T120408Z.jsonl`
- fusion on: `34.874597 tok/s` decode
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-combofuse-on-quad0123-p512n128-r2-20260506T120408Z.jsonl`

Full 4-card check, `-p 512 -n 512 -r 3`, fusion on:

- prompt: `86.274676 tok/s`
- decode: `34.842877 tok/s`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-combofuse-on-quad0123-p512n512-r3-20260506T120734Z.jsonl`

Conclusion: this is neutral. The short run was slightly better than its off control, but the full run did not beat the existing 4-card best of `34.929313 tok/s`. Keep the gates available for diagnostics, but do not treat them as a current best path.

## Vec4 F32 allreduce experiment

Goal: test whether vectorizing the tiny F32 allreduce kernels with `sycl::vec<float, 4>` improves the 3-card or 4-card decode path.

Temporary source gate:

```bash
GGML_SYCL_COMM_VEC4_F32=1
```

Results, `-p 512 -n 128 -r 2`:

- 3-card control off: `45.482621 tok/s`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-vec4-off-triple213-p512n128-r2-20260506T122210Z.jsonl`
- 3-card vec4 on: `45.313034 tok/s`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-vec4-on-triple213-p512n128-r2-20260506T122210Z.jsonl`
- 4-card control off: `34.890219 tok/s`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-vec4-off-quad2130-p512n128-r2-20260506T122210Z.jsonl`
- 4-card vec4 on: `34.198055 tok/s`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-vec4-on-quad2130-p512n128-r2-20260506T122210Z.jsonl`

The temporary `GGML_SYCL_COMM_VEC4_F32` source changes were removed after this negative result. Rebuild succeeded, and a post-revert 3-card smoke run produced `45.668525 tok/s` decode:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-smoke-after-vec4-revert-triple213-p512n128-r1-20260506T123929Z.jsonl`

Conclusion: simple F32 allreduce vectorization is not the bottleneck and hurts the 4-card path. Do not revisit scalar/vector formatting of this small allreduce unless a profiler shows it has become dominant after larger graph changes.

## Next direction

The 4-card regression is still likely structural: the current tensor-parallel path launches many small per-token collectives and does the output projection and meta allreduce as separate stages. The next useful Q4 direction is either:

- a true output-projection plus allreduce epilogue that avoids materializing and then reducing independent F32 partials; or
- reducing the number of graph collectives in the decode path.

LocalMaxxing: not submitted. These are neutral/negative diagnostic screens, not new best model results.
