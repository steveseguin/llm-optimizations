# Q4 Rotate-Root And MiniMax Follow-Up

Date: 2026-05-04

## Qwen3.6 Q4_0 Rotate-Root Allreduce

A small env-gated llama.cpp/SYCL diagnostic patch adds `GGML_SYCL_COMM_ROTATE_ROOT=1` for the experimental `GGML_SYCL_COMM_SINGLE_KERNEL=1` path. The default remains unchanged.

The patch rotates which backend queue launches each single-kernel allreduce. It was intended to test whether four-card negative scaling came from pinning all 128 decode allreduces/token to one root GPU.

Build note: the first rebuild failed at final link when oneAPI was not sourced. Re-running with `source /opt/intel/oneapi/setvars.sh --force` before `cmake --build ... --target llama-bench -j2` linked successfully.

| Topology | Selector | Prompt/Output | Rotate | tok/s out |
|---|---:|---:|---:|---:|
| 4x B70 | `0,1,2,3` | `512/32` | `0` | `30.493557` |
| 4x B70 | `0,1,2,3` | `512/32` | `1` | `30.728515` |
| 3x B70 | `2,1,3` | `512/128` | `0` | `42.573364` |
| 3x B70 | `2,1,3` | `512/128` | `1` | `41.359099` |
| 4x B70 | `0,1,2,3` | `512/128` | `0` | `32.164634` |
| 4x B70 | `0,1,2,3` | `512/128` | `1` | `31.753918` |

Conclusion: rotate-root is stable but not a speed path. Keep it only as a diagnostic env gate. It was not submitted to LocalMaxxing because it does not improve the recommended operating point and the 4-GPU negative-scaling result is already represented.

The `GGML_META_ALLREDUCE_STATS=2` traces also rule out a cheap metadata-delay fix for Qwen3.6: the reduction boundaries are direct `MUL_MAT` outputs (`linear_attn_out` / `attn_output` and `ffn_out`), not no-op view/reshape boundaries. The next Q4_0 path needs fused matmul/allreduce epilogues, reduce-scatter/all-gather style decomposition, or a lower-overhead tiny-message reduction primitive.

## MiniMax M2.7

Source review found:

- Tensor split is intentionally unsupported for `minimax-m2`.
- Layer split hits large contiguous SYCL device allocations.
- Row split reaches the intended split-buffer path but fails on `blk.12.ffn_down_exps.weight`, type `iq4_xs`, rows `[196608,393216)`, allocation `160432128` bytes.
- That allocation matches `196608` rows times `816` bytes/row for the `iq4_xs` expert tensor slice.
- Even after allocation, `GGML_OP_MUL_MAT_ID` with SYCL split buffers is not implemented and is likely to assert.

A four-B70 row-split `-ngl 11`, `-p 0 -n 1` confirmation run timed out after 240 seconds during setup with empty output:
`/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl11-p0n1-20260504T212415Z.*`.

Next viable patches:

- Low risk diagnostic: keep `MUL_MAT_ID` expert tensors off SYCL split buffers until split execution is implemented.
- Medium risk diagnostic: add host-USM fallback for failed split-buffer tensor allocations.
- Real speed path: implement expert-aligned split `MUL_MAT_ID`, where selected experts run only on the owning B70 and outputs are assembled correctly.
