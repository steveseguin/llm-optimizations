# 2026-05-07 Q4_0 Q8-cache allreduce+ADD guard fix

## Summary

After the mixed-row MMVQ diagnostic work, the current three-B70 Qwen3.6 27B Q4_0 recipe regressed badly:

- pre-fix control: `p0/n256/r2`, `27.676519 tok/s`
- expected current-stack range: high `48-49 tok/s`

The issue was not the new mixed fusion path. That path was disabled. The root cause was an over-broad guard in `ggml_backend_sycl_comm_allreduce_add_tensor()`:

```cpp
if (g_ggml_sycl_q8_cache > 0) {
    return false;
}
```

That guard belongs on the experimental lower-level `MUL_MAT+allreduce+ADD` diagnostic path, because that path has unresolved Q8-cache lifetime issues. It does not belong on the already validated meta `allreduce+ADD` path. With the guard present, the graph still recognized `fused_add`, but runtime allreduce fell back to plain `backend` paths.

## Fix

Removed the Q8-cache rejection from `ggml_backend_sycl_comm_allreduce_add_tensor()`. The Q8-cache guard remains on `comm_mul_mat_allreduce_add`, which is still diagnostic and disabled for the best recipe.

## Validation

Meta trace after the fix:

- `backend+add`: normal projection residual allreduce paths
- `backend+getrows`: final logits allreduce plus `GET_ROWS`

Short decode-only control:

- command shape: `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`, `-ub 128`, f16 KV
- `p0/n256/r3`: `48.926449 tok/s`

Full validation:

- `p512/n512/r3`
- prompt: `194.121497 tok/s`
- decode: `49.552666 tok/s`
- computed total: `78.947383 tok/s`
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/regression-debug-20260507/tp3-fixed-full-p512n512-r3-20260507T005722Z.jsonl`
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/regression-debug-20260507/tp3-fixed-full-p512n512-r3-20260507T005722Z.log`

## LocalMaxxing

The detailed annotated payload returned HTTP 500, matching earlier LocalMaxxing behavior on long Q4 payloads.

The reduced core-metric payload was accepted:

- ID: `cmous57ci00lqld01a8x5azdq`
- output: `49.552666 tok/s`
- total: `78.947383 tok/s`
- accepted at: `2026-05-07T01:00:43.506Z`

## Decision

Keep this as the current Q4_0 TP3 recipe:

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
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1
GGML_SYCL_FUSE_MMVQ2=1
GGML_SYCL_FUSE_MMVQ2_SWIGLU=1
GGML_SYCL_FUSE_RMS_NORM_MUL=1
```

The low TP3 mixed-fusion A/B runs around `27 tok/s` are invalid for mixed-fusion evaluation. They measured the misplaced allreduce+ADD guard, not the mixed-row kernel.
