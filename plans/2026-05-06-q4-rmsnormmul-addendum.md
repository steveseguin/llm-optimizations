# 2026-05-06 Q4_0 RMS_NORM+MUL Addendum

## Result

The new opt-in SYCL graph fusion `GGML_SYCL_FUSE_RMS_NORM_MUL=1` is now part of the best Q4_0 GGUF stack.

Validated Qwen3.6 27B Q4_0 GGUF results, 512 prompt / 512 output:

- 1x B70 `SYCL2`: `24.960284 tok/s`
- 2x B70 `SYCL2/SYCL1`, tensor split `1/1`, `-ub 128`: `42.106013 tok/s`
- 3x B70 `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`, `-ub 128`: `49.366188 tok/s`

LocalMaxxing accepted the 3x result as `cmoujcois00esld01c5s6bwht`.

## Correctness

Greedy `llama-completion` output matched byte-for-byte with the new fusion off and on:

- SHA256: `f7254271342a273042f88b21af7267f2fe5a06340ba68a9fc765746090a645aa`
- A debug one-token run confirmed `418` RMS_NORM+scale-MUL fusion calls.

No weights, quantization, KV dtype, sampling, speculative decoding, or GPU power limits changed.

## Follow-Up

The remaining Q4_0 work should move to larger fused epilogues:

- fused output-projection plus allreduce/residual ADD;
- broader same-activation multi-GEMV groups;
- reduce the number of per-token collectives before revisiting equal 4x tensor split.

The four-card assist rerun with this fusion failed with Level Zero OOM during `MUL_MAT`, so 4x Q4_0 remains a scheduling/kernel investigation rather than the production path.
