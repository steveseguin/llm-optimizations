# MiniMax Row-Split Addendum

Date: 2026-05-04

## Finding

The MiniMax M2.7 UD-IQ4_XS four-B70 path is not blocked by command syntax anymore. It is blocked by the current llama.cpp SYCL row-split expert path.

The `-ncmoe` staircase shows that each B70 can absorb only about 12 to 13 GPU-resident expert layers before a small split expert allocation fails:

- `-ncmoe 13` fails at block `25`;
- `-ncmoe 26` fails at block `37`;
- `-ncmoe 38` fails at block `49`;
- `-ncmoe 50` fails at block `60`.

The failed allocations are all `129761280` bytes for `iq3_s` expert slices, alternating across devices. Earlier row-split diagnostics also showed `160432128` byte `iq4_xs` expert-slice failures. Those sizes are plausible split slices, so the failure is not bad shape math.

## Decision

Do not spend more time on MiniMax flag sweeps until code changes are ready. `-ncmoe 62` is only a load proof because it makes the experts CPU/file-backed, which is not useful on this low-RAM host.

## Implementation Plan

1. Review `ggml_backend_sycl_split_buffer_init_tensor()` and the allocation lifetime for split expert tensors.
2. Add diagnostics for per-device allocation totals and failure point by tensor type, layer, and owning device.
3. Prototype a safer expert allocation strategy only if it keeps expert tensors GPU-resident.
4. Implement or prototype `GGML_OP_MUL_MAT_ID` on SYCL split buffers. The likely performance path is expert-aligned: selected experts execute where their rows live, then outputs are assembled or reduced with minimal cross-device traffic.
5. Re-test MiniMax only after the code path can keep a meaningful number of expert layers on the B70s and pass a `-p 0 -n 1` generation smoke.

## LocalMaxxing Policy

Do not submit the MiniMax staircase to LocalMaxxing as a benchmark. It is a useful engineering finding, but it produced no valid throughput number.
