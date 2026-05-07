# 2026-05-07 Q4_0 Four-Card Assist Refresh

## Context

After fixing the over-broad Q8-cache guard that disabled the validated `allreduce+ADD` path, I reran the best known four-card Q4_0 assist split. Earlier, the four-card assist layout reached `39.204149 tok/s`; equal four-card split was only `34.929313 tok/s`. The question was whether the current fused stack could make the assist layout useful again without changing model quality.

## Configuration

- Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
- Engine: local llama.cpp SYCL/Level Zero build, `db44417-local-sycl-q4-b70`
- Devices: `SYCL2/SYCL1/SYCL3/SYCL0`
- Split: tensor split `1/1/1/0.05`
- KV cache: f16
- Flash attention: enabled
- Microbatch: `-ub 128`
- Quality changes: none
- Power-limit changes: none

Environment:

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
GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2
```

## Results

| Run | Prompt | Output | Repeats | tok/s prompt | tok/s output | tok/s total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short screen | 0 | 128 | 2 | n/a | 42.782880 | n/a |
| Full validation | 512 | 512 | 3 | 117.882824 | 44.087560 | 64.174276 |

The full validation is:

- `+12.46%` versus the older four-card assist result (`39.204149 tok/s`)
- `+26.22%` versus equal four-card split (`34.929313 tok/s`)
- `-11.03%` versus the current three-card TP3 best (`49.552666 tok/s`)

## LocalMaxxing

The detailed annotated payload returned HTTP 500, matching the previous llama.cpp detailed-payload issue. The reduced payload was accepted:

- ID: `cmoute8kg00mbld017ye0dfbz`
- Output throughput: `44.08756 tok/s`
- Total throughput: `64.17427615741703 tok/s`

## Decision

This is the new best submitted four-card Q4_0 result. It proves the current fused stack and guard fix make the assist split viable again, but it still trails TP3. Four-card Q4_0 remains an investigation path; the next useful work is reducing communication and launch overhead rather than moving more rows onto the fourth GPU.

## Artifacts

- Short screen JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/q4-quad-assist005-rms-stack-p0n128-20260507T013139Z.jsonl`
- Full validation JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/q4-quad-assist005-rms-stack-p512n512-20260507T013303Z.jsonl`
