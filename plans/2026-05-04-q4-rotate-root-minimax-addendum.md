# Plan Addendum: Rotate-Root And MiniMax Follow-Up

Date: 2026-05-04

This addendum extends `plans/q4_0-gguf-b70-optimization-plan.md` with the latest follow-up work. The local working copy of the main plan has the same items appended as Track G items 23 and 24.

## Q4_0 Rotate-Root Allreduce

Status: tested, stable, diagnostic-only.

A new env-gated `GGML_SYCL_COMM_ROTATE_ROOT=1` patch rotates which backend queue launches the single-kernel allreduce under `GGML_SYCL_COMM_SINGLE_KERNEL=1`. Default behavior is unchanged when the env var is unset.

Results:

| Topology | Selector | Prompt/Output | Rotate | tok/s out |
|---|---:|---:|---:|---:|
| 4x B70 | `0,1,2,3` | `512/32` | `0` | `30.493557` |
| 4x B70 | `0,1,2,3` | `512/32` | `1` | `30.728515` |
| 3x B70 | `2,1,3` | `512/128` | `0` | `42.573364` |
| 3x B70 | `2,1,3` | `512/128` | `1` | `41.359099` |
| 4x B70 | `0,1,2,3` | `512/128` | `0` | `32.164634` |
| 4x B70 | `0,1,2,3` | `512/128` | `1` | `31.753918` |

Decision: root rotation is not a speed path. The four-card problem remains the count and cost of 128 small reductions per token.

Implementation direction: stop testing simple root/topology variants for Q4_0 4-GPU speed unless a profiler identifies a specific root-side bottleneck. Next useful implementation work is fused matmul/allreduce epilogues, reduce-scatter/all-gather-style decomposition, or a lower-overhead tiny reduction primitive.

## MiniMax M2.7

Status: source diagnosis complete; no valid performance number yet.

Findings:

- `LLAMA_SPLIT_MODE_TENSOR` is intentionally unsupported for `minimax-m2`.
- Row split reaches the SYCL split-buffer path and fails first on `blk.12.ffn_down_exps.weight`, type `iq4_xs`, rows `[196608, 393216)`, allocation `160432128` bytes.
- The failed allocation matches expected row-slice size, so the range math appears correct.
- `GGML_OP_MUL_MAT_ID` with SYCL split buffers is not implemented and is likely to assert even after allocation is handled.
- A four-B70 row-split `-ngl 11`, `-p 0 -n 1` confirmation timed out after 240 seconds with empty output, so no MiniMax benchmark should be submitted yet.

Next patches:

- Low-risk diagnostic: keep `MUL_MAT_ID` expert tensors off SYCL split buffers until split execution is implemented.
- Medium-risk diagnostic: add host-USM fallback for failed split-buffer tensor allocations.
- Real speed path: implement expert-aligned split `MUL_MAT_ID`, where selected experts run only on the owning B70 and outputs are assembled correctly.
