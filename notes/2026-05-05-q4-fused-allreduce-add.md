# Qwen3.6 Q4_0 Fused Allreduce + Residual Add

Date: 2026-05-05

## Summary

Implemented an env-gated Meta/SYCL fused collective for Q4 tensor-parallel decode:

- `GGML_META_FUSE_ALLREDUCE_ADD=1`
- new backend proc: `ggml_backend_comm_allreduce_add_tensor`
- SYCL implementation uses the existing single-kernel allreduce peer-read/write style
- pattern: immediate `ADD(partial, mirrored)` or `ADD(mirrored, partial)`

The fused kernel computes:

```text
sum(partial shards) + mirrored_residual
```

This preserves quality because it is equivalent to the baseline allreduce followed by residual ADD. It does not change model weights, quantization, KV dtype, sampling, speculative decode, or GPU power.

## Diagnostics

- one-site dual-card smoke with `GGML_META_FUSE_ALLREDUCE_ADD_LIMIT=1`: exactly one `path=backend+add`;
- all-site dual-card one-token diagnostic: `79` of `128` per-token reductions used `path=backend+add`;
- the fused sites were mostly `ffn_out-*`, plus attention-output sites whose graph shape has an immediate residual add.

## Validation

Command:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -sm tensor -ts 1/1/1 \
  -p 512 -n 512 -r 3 -ngl 99 -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 --poll 50 -o jsonl
```

Results:

- prompt: `135.665459 tok/s`
- decode samples: `43.9985`, `44.0192`, `43.9953 tok/s`
- average decode: `44.004344 tok/s`
- total throughput: `66.453788 tok/s`
- validation JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-fuseadd-validate-triple213-p512n512-r3-20260505T025706Z.jsonl`

Prior Q4_0 TP3 event-barrier validation was `43.605046 tok/s`, so this is a small but repeatable quality-preserving improvement.

## LocalMaxxing

- accepted reduced payload: `cmos1jmsv000iih04iifehc8d`
- full payloads with detailed `engineFlags` returned HTTP 500, matching the earlier full-payload issue.

## Artifacts

- patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-fused-allreduce-add-20260505.patch`
- patch sha256: `c4322255e33462a799d9bca1a33da66db7cafa9e7e37dd55c415d85a7c38cbf0`
- focused patch against the prior follow-up patch state: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-meta-fused-allreduce-add-focused-20260505.patch`
- focused patch sha256: `16d197831a426fa461c035fe90db972ebe2a54d9ef8d6b14b319539551bfa427`

## Follow-Ups

- test fused-add on 4x with the same fast command shape;
- investigate why only 79 of 128 reductions match the immediate residual pattern;
- for the remaining reductions, consider either a lower-level matmul/allreduce epilogue or a graph pattern that safely handles non-immediate residual structure.
