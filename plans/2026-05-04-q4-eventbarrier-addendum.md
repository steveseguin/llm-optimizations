# Q4 Event-Barrier Addendum

Date: 2026-05-04

## Status Change

The Q4_0 three-B70 path improved from the previous 512-output validation of `41.659 tok/s` to `43.605 tok/s` by replacing non-root allreduce marker tasks with event barriers in the experimental single-kernel allreduce path.

This is quality-preserving and software-only.

## Implementation Boundary

Patch gate:

`GGML_SYCL_COMM_EVENT_BARRIER=1`

The patch is intentionally narrow:

- same root allreduce kernel;
- same peer-read sum and writeback;
- same Q4_0 weights;
- same f16 KV cache;
- same tensor split;
- no speculative decode;
- no power limit changes.

Only non-root queue synchronization changed from a dependent `single_task` marker to `ext_oneapi_submit_barrier({reduce})`.

## Next Steps

1. Run the same gate on 2x B70 at 512 output.
2. Run the same gate on 4x B70 at 128 and 512 output.
3. If 4x remains around the low 30 tok/s range, stop spending time on root ordering or marker tasks. The remaining 4x bottleneck is almost certainly reduction frequency/fanout.
4. Start a fused output path for the 20 KB F32 allreduce tensors:
   - first target direct `MUL_MAT` outputs for `linear_attn_out` / `attn_output` and `ffn_out`;
   - avoid graph-wide semantic changes until the fused epilogue proves correctness on one allreduce site;
   - use existing allreduce trace logs to identify the repeated tensor names.

## Sharing

LocalMaxxing result:

`cmortp5vn000el404dj3zqv0u`

The API accepted a reduced payload. Full command and flags are stored in the data artifact and local payload log.
