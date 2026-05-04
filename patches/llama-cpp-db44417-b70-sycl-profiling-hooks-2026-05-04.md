# llama.cpp SYCL Profiling Hooks for B70 Qwen3.6 Work

Date: 2026-05-04 UTC

Worktree: `/home/steve/src/llama.cpp-q4-b70`

Base: upstream llama.cpp `db44417` plus the earlier B70/Qwen3.6 SYCL tensor-split and single-kernel allreduce patches.

This note captures the incremental profiling/debug hooks added after the initial allreduce patch. These hooks are diagnostic and should remain env-gated.

## Meta Allreduce Trace and Timing

File: `ggml/src/ggml-backend-meta.cpp`

Added env var:

- `GGML_META_ALLREDUCE_STATS=1`: print aggregate count/bytes by allreduce size.
- `GGML_META_ALLREDUCE_STATS=2`: also print tensor name/op/type/shape for each allreduce.
- `GGML_META_ALLREDUCE_STATS=3`: synchronize before and after each allreduce and print timing. This intentionally perturbs runtime.

Important behavior:

- Output goes to `stderr`, not `GGML_LOG_INFO`, so `llama-bench -o jsonl` does not hide the diagnostics.
- The timing path calls `ggml_backend_synchronize()` on every simple backend before and after the allreduce. Use it only on one-token traces.

The hook identified 128 allreduces per decode token for Qwen3.6 27B Q4_0 tensor split: one attention-output reduction and one FFN-output reduction per layer, all 20 KiB F32.

## SYCL Op Timing

File: `ggml/src/ggml-sycl/ggml-sycl.cpp`

Added env var:

- `GGML_SYCL_OP_STATS=1`: collect cumulative explicit-sync timing by `ggml_op`.
- `GGML_SYCL_OP_STATS=2`: also print per-node timing lines.

Important behavior:

- The hook waits the active stream after each node, so it changes runtime behavior.
- Use with `GGML_SYCL_DISABLE_GRAPH=1` and short one-token runs.
- Output goes to `stderr`.

Warm single-B70 one-token profile showed `MUL_MAT` dominance:

- `MUL_MAT`: 1058 calls, `585.751 ms` total, `553.640 us` average.
- `GATED_DELTA_NET`: `2.131 ms` total.
- `SSM_CONV`: `2.061 ms` total.

This shifted single-card work away from recurrent-kernel fusion and back toward Q4_0 reordered MMVQ/dataflow.

## MMVQ Launch Constant Hook

Files:

- `ggml/src/ggml-sycl/presets.hpp`
- `ggml/src/ggml-sycl/mmvq.cpp`

Added compile-time macro:

```cpp
#ifndef GGML_SYCL_REORDER_MMVQ_SUBGROUPS
#define GGML_SYCL_REORDER_MMVQ_SUBGROUPS WARP_SIZE
#endif
```

The reordered Q4_0 MMVQ path now uses this macro instead of a hard-coded `WARP_SIZE` subgroup count.

Tested build:

```bash
-DCMAKE_CXX_FLAGS=-DGGML_SYCL_REORDER_MMVQ_SUBGROUPS=8
```

Result:

- `24.410 tok/s` on single B70 Q4_0, 512 prompt / 256 output shape.
- This was a regression versus the default path, so keep the default.

## Current Local Diff Scope

`git diff --stat` in `/home/steve/src/llama.cpp-q4-b70` after these additions:

```text
ggml/include/ggml-sycl.h             |   2 +-
ggml/src/ggml-backend-meta.cpp       |  89 +++++-
ggml/src/ggml-sycl/CMakeLists.txt    |  14 +-
ggml/src/ggml-sycl/ggml-sycl.cpp     | 554 ++++++++++++++++++++++++++++++++---
ggml/src/ggml-sycl/mmvq.cpp          |   3 +-
ggml/src/ggml-sycl/presets.hpp       |   3 +
ggml/src/ggml-vulkan/ggml-vulkan.cpp |  12 +-
src/llama-model.cpp                  |  23 +-
```

The full local diff is cumulative and includes the earlier experimental SYCL tensor-split, allreduce, Vulkan B70, and Qwen recurrent split-anchor changes. The earlier patch artifact is `patches/llama-cpp-db44417-b70-qwen36-sycl-single-kernel-allreduce.md`; this file documents the newer profiling additions and negative MMVQ launch-constant result.

## Rebuild Command

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j2
```

The AOT SYCL link invokes `ocloc` and can take several minutes even for small host-side edits.
