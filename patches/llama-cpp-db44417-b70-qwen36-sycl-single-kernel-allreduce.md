# llama.cpp B70/Qwen3.6 SYCL Patch Set

Base worktree: `/home/steve/src/llama.cpp-q4-b70`

Base upstream commit: `db44417`

Generated diff locally with:

```bash
git -C /home/steve/src/llama.cpp-q4-b70 diff -- \
  ggml/include/ggml-sycl.h \
  ggml/src/ggml-backend-meta.cpp \
  ggml/src/ggml-sycl/CMakeLists.txt \
  ggml/src/ggml-sycl/ggml-sycl.cpp \
  ggml/src/ggml-vulkan/ggml-vulkan.cpp \
  src/llama-model.cpp
```

Local diff stat at the time of this note:

```text
ggml/include/ggml-sycl.h             |   2 +-
ggml/src/ggml-backend-meta.cpp       |  48 ++++-
ggml/src/ggml-sycl/CMakeLists.txt    |  14 +-
ggml/src/ggml-sycl/ggml-sycl.cpp     | 403 +++++++++++++++++++++++++++++++----
ggml/src/ggml-vulkan/ggml-vulkan.cpp |  12 +-
src/llama-model.cpp                  |  23 +-
6 files changed, 451 insertions(+), 51 deletions(-)
```

## Reproduction Notes

These are experimental patches, not an upstream-ready cleaned series. Keep every behavior behind its env gate while testing.

Build command used locally:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j2
```

Best validated run:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_GRAPH=0 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_DISABLE_OPT=0 \
GGML_SYCL_PRIORITIZE_DMMV=0 \
GGML_SYCL_ASYNC_PEER_COPY=0 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  --prio 0 -dev SYCL0/SYCL1/SYCL2 -ngl 99 -p 0 -n 512 \
  -sm tensor -ts 1/1/1 -b 512 -ub 32 -ctk f16 -ctv f16 \
  -t 8 --poll 50 -fa 1 -r 3 -o jsonl
```

Result: `41.736585 tok/s`, samples `41.6977`, `41.6966`, `41.8155`.

## Patch Components

### `ggml/src/ggml-sycl/CMakeLists.txt`

Purpose: make Intel GPU AOT builds work for `intel_gpu_bmg_g31`.

Changes:

- For Intel GPU `GGML_SYCL_DEVICE_ARCH` values matching `intel_gpu_*`, use `-fsycl-targets=${GGML_SYCL_DEVICE_ARCH}` instead of `-Xsycl-target-backend --offload-arch=...`.
- Skip `-Xs -ze-intel-greater-than-4GB-buffer-required` for Intel GPU AOT target strings.

### `src/llama-model.cpp`

Purpose: fix Qwen recurrent split planning for 3-way tensor split.

Changes:

- Detect Qwen recurrent families: `LLM_ARCH_QWEN3NEXT`, `LLM_ARCH_QWEN35`, `LLM_ARCH_QWEN35MOE`.
- For recurrent layers, anchor these tensors to `ssm_out.weight`:
  - `attn_qkv.weight`
  - `attn_qkv.bias`
  - `attn_gate.weight`

Reason:

Qwen recurrent cache/SSM paths use `ssm_out.weight` split granularity. QKV/gate tensors anchored to their own Q4_0 block plan caused `ggml-backend-meta.cpp:1014` split-ratio aborts for equal 3-way tensor split.

### `ggml/src/ggml-backend-meta.cpp`

Purpose: better diagnostics and allreduce instrumentation.

Changes:

- Replace the opaque split-ratio assert with a detailed `GGML_ABORT` that prints tensor/op/source names, axes, sizes, and lhs/rhs values.
- Add env `GGML_META_ALLREDUCE_STATS`:
  - `1`: per-graph summary of allreduce count and bytes.
  - `2`: per-allreduce detail plus summary.

Observed diagnostic:

```text
split ratio mismatch in conv_input-0[CONCAT] src 1 qkv_mixed_transposed-0[TRANSPOSE]: dst_axis=1 dst_ne=10240 split_ne=2560 src_axis=1 src_ne=10240 src_sum=3200 lhs=26214400 rhs=32768000
```

Observed allreduce pattern after fixes:

- 128 F32 allreduces per generated token.
- Each allreduce is 5120 F32 elements, 20 KiB.
- Remaining bottleneck is many tiny reductions and synchronization, not raw data volume.

### `ggml/include/ggml-sycl.h`

Purpose: make SYCL split buffers main-device aware.

Change:

```cpp
ggml_backend_sycl_split_buffer_type(int main_device, const float * tensor_split)
```

This lets split buffer types be keyed by owning main device instead of globally reusing a single split buffer type.

### `ggml/src/ggml-sycl/ggml-sycl.cpp`

Purpose: enable Qwen tensor-split correctness and reduce Meta allreduce overhead.

Key env gates:

- `GGML_SYCL_ASYNC_PEER_COPY`, default `0`.
- `GGML_SYCL_ASYNC_CPY_TENSOR`, default `0`.
- `GGML_SYCL_COMM_ALLREDUCE`, default `0`.
- `GGML_SYCL_COMM_SINGLE_KERNEL`, default `0`.

Correctness/split changes:

- Split buffer context now tracks `main_device` and has per-device streams.
- Split buffer type map is keyed by `(main_device, tensor_split)`.
- SYCL split buffer `supports_buft` only returns true for the owning backend device.
- Reorder optimization is disabled for SYCL split buffers.
- Split matmul uses per-device split weight pointers:
  - `src0_extra->data_device[i]` when the source tensor is split.
- `SSM_CONV` and `GATED_DELTA_NET` do not accept split-buffer inputs until their kernels are made split-aware.

Async copy changes:

- `GGML_SYCL_ASYNC_CPY_TENSOR=1` registers `.cpy_tensor_async`.
- Cross-device copies insert a source stream barrier and enqueue destination stream memcpy with an event dependency instead of relying on blocking waits.

Meta comm hook:

- `GGML_SYCL_COMM_ALLREDUCE=1` registers:
  - `ggml_backend_comm_init`
  - `ggml_backend_comm_free`
  - `ggml_backend_comm_allreduce_tensor`
- Supported scope is 2-4 SYCL backends and contiguous `GGML_TYPE_F32` tensors.
- Fallback remains available when tensor shape/type/buffer constraints are not met.

Single-kernel allreduce:

- `GGML_SYCL_COMM_SINGLE_KERNEL=1` uses backend 0 as root.
- Root queue waits for all backend ready events.
- Root kernel peer-reads up to four partial tensors, sums them, and writes the result back to all participating device allocations.
- Each non-root backend stream receives a dependency marker event after the root reduction.
- This path is validated by `tools/sycl-peer-read-test.cpp`, but it remains experimental and env-gated.

### `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Purpose: keep Vulkan B70 experiments reproducible.

Changes:

- Map PCI ID `0xE223` to 32 Intel Xe2 shader cores.
- Add `GGML_VK_INTEL_XE2_DMMV_LARGE_MAX_M` to sweep the large-DMMV threshold for Intel Xe2.

## Validation Artifacts

Local result files:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-comm-single-kernel-dual03-512-reps3-20260504T040658Z.jsonl`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-comm-single-kernel-triple012-512-reps3-20260504T044231Z.jsonl`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-comm-single-kernel-triple213-512-reps3-20260504T045845Z.jsonl`
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-comm-single-kernel-quad0123-512-r1-20260504T044459Z.jsonl`

LocalMaxxing submissions:

- 2 B70 single-kernel: `cmoqp6jpq0004lb04241n9ns3`
- 3 B70 root `0,1,2`: `cmoqptj6i000blb04j0i2u9yo`
- 3 B70 root `2,1,3`: `cmoqqed6s0007jv049wnizwle`

## Current TODOs

- Convert this experimental patch set into a smaller upstreamable series.
- Investigate a 4-GPU hierarchical reduction; the current root fanout reaches only about `31 tok/s`.
- Explore whether the 128 per-token allreduces can be reduced, fused, or scheduled with less host/runtime overhead.
- Revisit single-card Linux SYCL performance versus the known Windows Q4_0 result above `27 tok/s`.
