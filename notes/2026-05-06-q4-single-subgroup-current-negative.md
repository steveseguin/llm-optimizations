# 2026-05-06 Q4_0 single-B70 subgroup runtime screen

## Goal

Check whether `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME` improves single-B70 Qwen3.6 27B Q4_0 decode after the current fused stack landed.

This is a quality-preserving runtime screen: same Q4_0 GGUF, f16 KV, no speculative decoding, no sampling change, and no power change.

## Configuration

- model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`;
- device: `SYCL2`;
- command shape: `-ngl 99 -sm none -fa 1 -ub 128 -ctk f16 -ctv f16 -t 8 -p 0 -n 256 -r 2 --poll 50`;
- env stack: `GGML_SYCL_Q8_CACHE=1`, `GGML_SYCL_DISABLE_DNN=1`, `GGML_SYCL_ASYNC_CPY_TENSOR=0`, `GGML_SYCL_FUSE_MMVQ2=1`, `GGML_SYCL_FUSE_MMVQ2_SWIGLU=1`, `GGML_SYCL_FUSE_RMS_NORM_MUL=1`.

## Results

| subgroup runtime | tok/s |
| --- | ---: |
| default | 24.930018 |
| 1 | 24.894307 |
| 16 | 24.893128 |
| 2 | 24.886190 |
| 8 | 24.876417 |
| 4 | 24.874003 |

Artifacts:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/single-subgroup-current-20260506/single-subgroup-current-p0n256-r2-20260506T225012Z.tsv`.

## Interpretation

The current default subgroup behavior remains best for single-B70 decode. This does not move us toward the Windows Q4_0 target, so do not set `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME` in the current best recipe.

The single-card gap now looks unlikely to be closed by simple subgroup-count runtime tuning. Continue with deeper Q4_0 matvec work: ESIMD/XMX kernel experiments, better activation reuse across projections, or lower-overhead fused Q4_0 MMVQ variants.

