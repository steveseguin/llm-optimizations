# Qwen3.6 Q4_0 Local-Write Collective Diagnostics

Date: 2026-05-05

## Summary

After the quality-preserving fused allreduce + residual-add patch improved the 3x B70 path, I tested two 4x collective variants to see whether root fanout or remote residual reads were the main limiter.

All tested variants were neutral or negative for decode. They remain useful diagnostics, but they should stay off for performance runs.

## Variants

- `GGML_SYCL_COMM_LOCAL_WRITE=1`
  - ordinary allreduce: each GPU gathers peer partials into local temp buffers, then writes only its local output;
  - fused allreduce+ADD: each GPU directly computes `sum(partials) + local_residual` into its local output.
- `GGML_SYCL_COMM_LOCAL_WRITE=2`
  - applies local-write only to fused allreduce+ADD sites;
  - ordinary allreduces stay on the existing root single-kernel path.
- `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1`
  - keeps the root single-kernel fused-add path;
  - uses the root residual for all outputs because the residual split state is mirrored.

## Results

Qwen3.6 27B Q4_0 GGUF, 4x B70 selector `0,1,2,3`, 512 prompt / 128 output, no explicit `-b`:

| Variant | Decode tok/s | JSONL |
| --- | ---: | --- |
| root fused-add baseline | `33.219955` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-fastshape-quad0123-fuseadd-p512n128-20260505T031139Z.jsonl` |
| local-write all-sites | `30.681785` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-localwrite-quad0123-fuseadd-p512n128-20260505T033050Z.jsonl` |
| local-write fused-only | `32.365769` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-localwrite2-fuseonly-quad0123-p512n128-20260505T033941Z.jsonl` |
| root residual reuse | `33.074515` | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-rootresid-quad0123-fuseadd-p512n128-20260505T034916Z.jsonl` |

## Conclusion

The 4x cliff is not solved by avoiding root residual reads or by replacing root writes with per-device local writes. For these 20 KiB reductions, extra copy/event/kernel overhead is enough to lose the decode path.

Next 4x Q4_0 work should target fewer synchronization points or a lower-level matmul/reduction epilogue, not more tiny collective topology variants.
