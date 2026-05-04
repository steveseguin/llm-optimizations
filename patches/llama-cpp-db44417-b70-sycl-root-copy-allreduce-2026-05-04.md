# llama.cpp SYCL Root-Copy Allreduce Experiment

Date: 2026-05-04
Base: llama.cpp `db44417` local B70 worktree
Status: experimental, env-gated, slower than current best path

## Purpose

The current best multi-GPU Q4_0 path uses an env-gated single-kernel F32 allreduce. It is good for 2-3 B70s but regresses with 4 B70s because backend 0 must peer-read and peer-write all reduced vectors across 128 small allreduces per generated token.

This experiment adds a middle path:

- reduce on backend 0 using remote reads;
- write the reduced vector only to backend 0 inside the kernel;
- broadcast the reduced 20 KiB vector back to peers using SYCL peer `memcpy` operations.

The hypothesis was that peer copies might beat scalar remote writes at four GPUs.

## New Env Flag

```bash
GGML_SYCL_COMM_ROOT_COPY=1
```

It only affects SYCL Meta allreduce when `GGML_SYCL_COMM_ALLREDUCE=1` is also enabled.

## Patch Sketch

Add a runtime flag near the other SYCL communication flags:

```cpp
int g_ggml_sycl_comm_root_copy = 0;
...
g_ggml_sycl_comm_root_copy = get_sycl_env("GGML_SYCL_COMM_ROOT_COPY", 0);
```

In `ggml_backend_sycl_comm_allreduce_tensor`, add:

```cpp
const bool use_root_copy = g_ggml_sycl_comm_root_copy && n_backends >= 2 && n_backends <= 4;
```

Skip temporary-buffer allocation for this path:

```cpp
if (sycl_ctx->device != buf_ctx->device ||
        (!use_single_kernel && !use_pairwise4 && !use_root_copy && !comm_ctx->ensure_tmp(i, n_reduce_steps*nbytes))) {
    return false;
}
```

Add the root-copy branch before the existing single-kernel branch:

```cpp
} else if (use_root_copy) {
    ggml_backend_sycl_context * sycl_ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    const queue_ptr stream0 = sycl_ctx0->stream(sycl_ctx0->device, 0);
    float * dst0 = (float *) tensors[0]->data;
    float * dst1 = (float *) tensors[1]->data;
    float * dst2 = n_backends > 2 ? (float *) tensors[2]->data : nullptr;
    float * dst3 = n_backends > 3 ? (float *) tensors[3]->data : nullptr;

    sycl::event reduce = stream0->submit([=](sycl::handler & h) {
        h.depends_on(ready);
        h.parallel_for(sycl::range<1>((size_t) ne), [=](sycl::id<1> idx) {
            const size_t k = idx[0];
            float sum = dst0[k] + dst1[k];
            if (n_backends > 2) {
                sum += dst2[k];
            }
            if (n_backends > 3) {
                sum += dst3[k];
            }
            dst0[k] = sum;
        });
    });

    for (size_t i = 1; i < n_backends; ++i) {
        ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *) comm_ctx->backends[i]->context;
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        void * dst = tensors[i]->data;
        stream->submit([=](sycl::handler & h) {
            h.depends_on(reduce);
            h.memcpy(dst, dst0, nbytes);
        });
    }

    return true;
```

## Validation

Correct multi-device CLI syntax matters. Use slash-separated devices:

```bash
-dev SYCL0/SYCL1/SYCL2
```

Do not use comma-separated devices. A comma form caused Level Zero OOM in `MUL_MAT` before the allreduce path was evaluated.

Q4_0 GGUF root-copy scaling, 128 generated tokens:

| Mode | Selector | Result |
| --- | --- | ---: |
| 2x B70 | `level_zero:0,3` | `38.259 tok/s` |
| 3x B70 | `level_zero:2,1,3` | `39.817 tok/s` |
| 4x B70 | `level_zero:3,0,1,2` | `30.371 tok/s` |

Artifact: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-root-copy-scaling-n128-20260504T142851Z.tsv`.

## Outcome

Root-copy is stable but slower than the existing single-kernel allreduce path:

- best 2x single-kernel: `39.849 tok/s`;
- best 3x single-kernel: `41.737 tok/s`;
- 4x single-kernel: `31.482 tok/s`.

Decision: keep this patch only as a diagnostic branch. Four-card speed needs fewer synchronization points or a persistent/fused communication scheme, not another simple root topology.
