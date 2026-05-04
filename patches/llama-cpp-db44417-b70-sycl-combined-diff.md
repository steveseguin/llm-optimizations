# llama.cpp db44417 B70 SYCL Combined Diff

Date: 2026-05-04

This artifact captures the current local diff in `/home/steve/src/llama.cpp-q4-b70` that produced the B70 Q4_0 optimization results.

Compressed patch artifact:

- `patches/llama-cpp-db44417-b70-sycl-combined.diff.gz.b64`

Decode and apply from a clean llama.cpp checkout at commit `db44417`:

```bash
base64 -d patches/llama-cpp-db44417-b70-sycl-combined.diff.gz.b64 | gunzip > /tmp/llama-cpp-db44417-b70-sycl-combined.diff
git -C /path/to/llama.cpp checkout db44417
git -C /path/to/llama.cpp apply /tmp/llama-cpp-db44417-b70-sycl-combined.diff
```

The diff includes the local experimental work used during the B70 sessions:

- SYCL tensor split main-device API adjustment.
- Qwen recurrent-family split anchoring for QKV/gate tensors.
- SYCL async tensor copy path.
- SYCL Meta comm hooks.
- direct/single-kernel/root-copy/pairwise allreduce experiments.
- graph-scoped Q8_1 activation cache controlled by `GGML_SYCL_Q8_CACHE`.
- MUL_MAT and allreduce diagnostic timing flags.
- BMG-G31 AOT CMake flag fixes.
- Intel B70 Vulkan shader-core detection and DMMV threshold knob.

Primary runtime flags for the best Q4_0 results:

```bash
GGML_SYCL_Q8_CACHE=1
GGML_SYCL_ASYNC_CPY_TENSOR=1
GGML_SYCL_COMM_ALLREDUCE=1
GGML_SYCL_COMM_SINGLE_KERNEL=1
```

The patch is not upstream-ready as-is. It is preserved for reproducibility of the measured B70 optimization path and for future cleanup into smaller reviewable changes.
