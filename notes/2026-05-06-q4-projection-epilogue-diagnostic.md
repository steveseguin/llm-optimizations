# 2026-05-06 Q4_0 projection epilogue diagnostic

## Goal

Probe whether the Q4_0 row-parallel projection boundary can be scheduled as `MUL_MAT -> allreduce -> ADD` inside one backend helper instead of computing the `MUL_MAT` in the normal per-backend graph and then launching the existing fused allreduce+ADD helper.

This is a first diagnostic step toward a real projection GEMV plus allreduce/residual epilogue. It is not a fused math kernel yet.

## Patch

Added an off-by-default meta-backend/SYCL helper:

- env gate: `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=1`;
- backend proc: `ggml_backend_comm_mul_mat_allreduce_add_tensor`;
- path label: `backend+mulmat+add`;
- target pattern: Q4_0 `MUL_MAT`, F32 activations, F32 partial output, followed by an existing fused allreduce+ADD residual;
- backend limit: SYCL only, 2 to 4 backends, contiguous F32 residual/output sizes matching the projection partial;
- Q8 safety guard: if `GGML_SYCL_Q8_CACHE` is nonzero, the meta planner does not form this fusion, and the SYCL helper also declines.

The Q8 guard matters. The first Q8-on probe exposed that planning the fusion and then falling back after helper decline still computed the skipped `MUL_MAT` through the aux-node path, bypassing the normal graph-level Q8 cache lifetime handling. That ended with:

```text
GGML_ASSERT(pool_size == 0) failed
```

Moving the guard into the meta planner restored the normal Q8-enabled schedule.

Patch artifact:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-mulmat-add-diagnostic-current-20260506.patch.gz.b64`
- repo copy: `patches/llama-cpp-sycl-meta-mulmat-add-diagnostic-current-20260506.patch.gz.b64`

## Validation

Q8-on planner guard smoke:

- command shape: 3x B70, `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`, `p0/n1/r1`, current fused Q4_0 stack;
- env included `GGML_SYCL_Q8_CACHE=1` and `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=1`;
- result: no abort, no `backend+mulmat+add` path, `backend+getrows` still active;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-on-planner-guard-p0n1-20260506T224039Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-on-planner-guard-p0n1-20260506T224039Z.log`.

Q8-off path smoke:

- env included `GGML_SYCL_Q8_CACHE=0` and `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=1`;
- result: 142 `backend+mulmat+add` stats entries, 2 `backend+getrows` entries, no assertions;
- one-token speed: `42.879940 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-off-mulmat-add-p0n1-20260506T224200Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-off-mulmat-add-p0n1-20260506T224200Z.log`.

Q8-off short decode A/B, `p0/n128`:

- scheduler hook off: `48.239722 tok/s`;
- scheduler hook on: `47.700182 tok/s`;
- hook-on delta: `-0.539540 tok/s`, about `-1.12%`.

Artifacts:

- off JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-off-no-mulmat-add-p0n128-r3-20260506T224315Z.jsonl`;
- on JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/proj-epilogue-20260506/q8-off-mulmat-add-p0n128-r3-20260506T224437Z.jsonl`.

## Interpretation

The scheduler hook works as a path probe, but the current implementation is not a speed win. It probably adds scheduling overhead without removing the expensive kernel boundary we actually care about. A useful next version needs to move lower: combine the Q4_0 MMVQ work, cross-device reduction, and residual epilogue more directly, or group same-activation projections before the collective.

Do not enable `GGML_META_FUSE_MUL_MAT_ALLREDUCE_ADD=1` in the current best Q4_0 recipe. It is not LocalMaxxing-worthy because it was tested only with Q8 cache disabled and regressed the short decode A/B.

