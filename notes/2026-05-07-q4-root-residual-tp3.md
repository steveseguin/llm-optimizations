# 2026-05-07 Q4_0 TP3 Root-Residual Refresh

## Context

After the Q8-cache guard fix restored the validated fused `allreduce+ADD` path, I screened small collective-side options that could remove residual-buffer reads without changing model weights, KV dtype, sampling, or GPU power. The useful option was `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1` on the three-card TP3 layout.

## Configuration

- Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
- Engine: local llama.cpp SYCL/Level Zero build, `db44417-local-sycl-q4-b70`
- Devices: `SYCL2/SYCL1/SYCL3`
- Split: tensor split `1/1/1`
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
GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1
GGML_META_FUSE_ALLREDUCE_ADD=1
GGML_META_FUSE_ALLREDUCE_GET_ROWS=1
GGML_SYCL_FUSE_MMVQ2=1
GGML_SYCL_FUSE_MMVQ2_SWIGLU=1
GGML_SYCL_FUSE_RMS_NORM_MUL=1
```

## Results

| Run | Prompt | Output | Repeats | tok/s prompt | tok/s output | tok/s total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP3 short baseline | 0 | 128 | 2 | n/a | 48.349109 | n/a |
| TP3 short root-residual | 0 | 128 | 2 | n/a | 48.894885 | n/a |
| TP3 full root-residual | 512 | 512 | 3 | 193.705831 | 50.808572 | 80.501734 |

This is `+2.53%` versus the guard-fix TP3 validation (`49.552666 tok/s`) and is the current best quality-preserving Q4_0 GGUF result.

## LocalMaxxing

The reduced payload was accepted:

- ID: `cmouvurhh00nqld010dtr4xrl`
- Output throughput: `50.808572 tok/s`
- Total throughput: `80.5017335619557 tok/s`

## Negative Follow-Ups

| Screen | Best / comparison | Decision |
| --- | ---: | --- |
| Four-card assist root-residual full validation | `43.884730 tok/s` versus accepted `44.087560 tok/s` | Keep root-residual off for 4x assist. |
| Q4_1 forced MMVQ | `42.344754 tok/s` versus default DMMV `42.552518 tok/s` | Default-off experiment only. |
| Q8-off `MUL_MAT+allreduce+ADD` diagnostic | `41.707508 tok/s` versus Q8-cache-on control `42.732977 tok/s` | Keep Q8 cache on; leave diagnostic off. |
| Four-card root skip / rotation | short screens at or below baseline except one non-durable root-residual screen | Do not pursue as a standalone topology tweak. |

## Correctness Status

This result does not intentionally change model quality: same Q4_0 weights, f16 KV, greedy benchmark mode, no speculative decoding, no sampling change, and no power-limit change. The root-residual flag reuses the mirrored residual tensor on the root device in the fused `allreduce+ADD` path.

The attempted `llama-cli` text smoke was inconclusive. The first two commands had CLI syntax errors, and the corrected command hung while producing a large repeated stdout file. Before upstreaming or treating this as a final correctness proof, add a lower-overhead token/logit comparison harness.

## Artifacts

- TP3 short A/B TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/comm-micro-sweep/q4-tp3-fuseadd-root-p0n128-20260507T023419Z.tsv`
- TP3 full validation JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/comm-micro-sweep/fuseadd_root-p512n512-r3-20260507T023649Z.jsonl`
- TP3 full validation log: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/comm-micro-sweep/fuseadd_root-p512n512-r3-20260507T023649Z.log`
- Four-card comm micro-sweep TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/comm-micro-sweep/q4-quad-assist005-comm-micro-p0n128-20260507T022429Z.tsv`
- Four-card root-residual full JSON: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/comm-micro-sweep/fuseadd_root-p512n512-r3-20260507T023158Z.jsonl`
- Q4_1 MMVQ A/B TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/q4_1-mmvq-ab/q4-quad-assist005-q4_1-mmvq-ab-p0n128-20260507T021728Z.tsv`
- Projection epilogue diagnostic TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/quad-assist-refresh-20260507/mulmat-allreduce-fuse/q4-quad-assist005-mulmat-allreduce-fuse-p0n128-20260507T022028Z.tsv`
- Inconclusive correctness attempt directory: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fuseadd-root-residual-tp3-20260507T024037Z`
