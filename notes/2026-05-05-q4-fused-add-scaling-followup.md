# Qwen3.6 Q4_0 Fused-Add Scaling Follow-Up

Date: 2026-05-05

## Summary

After validating the 3x B70 fused allreduce + residual-add path at `44.004344 tok/s`, I screened the same patch on 4x and 2x layouts.

The patch is quality-preserving, but the speed benefit is topology-specific:

- 3x selector `2,1,3`: small repeatable improvement;
- 4x selector `0,1,2,3`: small improvement, still poor absolute scaling;
- 2x selector `0,3`: neutral on the known-good dual command shape.

## Results

| Layout | Selector | Batch shape | Control decode | Fused-add decode | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| 4x tensor | `level_zero:0,1,2,3` | no explicit `-b`, `n_batch=2048` | `32.383337 tok/s` | `33.219955 tok/s` | +2.6%, still below 3x |
| 2x tensor | `level_zero:0,3` | explicit `-b 512`, `n_batch=512` | `40.278630 tok/s` | `40.265194 tok/s` | neutral |

Artifacts:

- 4x control: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-fastshape-quad0123-base-p512n128-20260505T031139Z.jsonl`
- 4x fused-add: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-fastshape-quad0123-fuseadd-p512n128-20260505T031139Z.jsonl`
- 2x control: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-bestshape-dual03-base-b512-p512n256-20260505T031512Z.jsonl`
- 2x fused-add: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-bestshape-dual03-fuseadd-b512-p512n256-20260505T031512Z.jsonl`

## Interpretation

The current single-kernel allreduce path uses one root device to read all partial shards and write mirrored output tensors back to every device. On four B70s, that root fanout likely becomes the dominant cost for the tiny per-token F32 reductions.

The next Q4_0 target is therefore not more batch or selector sweeping. It is a software collective variant that reduces remote write fanout, for example:

- per-device local-write kernels, where each GPU reads the peer partials and writes only its own mirrored output;
- hierarchical 2+2 reduction and broadcast, if it reduces the worst cross-root traffic;
- a lower-level output-projection epilogue that fuses row-parallel matmul, reduction, and residual add before materializing the mirrored tensor.

No new LocalMaxxing submission was made for this follow-up because the 4x result is still worse than the existing submitted 3x and FP8 records, and the dual result is neutral.
