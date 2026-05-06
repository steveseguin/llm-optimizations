# Q4_0 RMS_NORM + Scale-MUL Fusion

Date: 2026-05-06

## Goal

Reduce per-token launch and memory traffic in Qwen3.6 27B `Q4_0` GGUF without changing model weights, quantization, KV dtype, sampling, speculative decoding, or GPU power settings.

## Patch

Added an opt-in SYCL graph fusion behind `GGML_SYCL_FUSE_RMS_NORM_MUL=1`.

- `ggml/src/ggml-sycl/norm.hpp`: declared `ggml_sycl_op_rms_norm_mul`.
- `ggml/src/ggml-sycl/norm.cpp`: added a fused RMSNorm kernel that applies the 1D F32 scale vector while writing the final MUL output.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: added the env gate, graph matcher, and conservative safety checks.
- `ggml/src/ggml-backend-meta.cpp` and `ggml/src/ggml-alloc.c`: added allocator diagnostics so failed meta compute-buffer reserves report the failing simple backend and size instead of falling through to an assertion.

The matcher only accepts F32 `RMS_NORM -> MUL` where the other MUL input is a contiguous 1D F32 scale tensor on the same non-split SYCL device buffer. It is disabled when explicit op stats are enabled.

Patch artifact: `patches/llama-cpp-sycl-rmsnormmul-current-20260506.patch.gz.b64`.

## Correctness

Greedy `llama-completion` output matched byte-for-byte with the RMS fusion off and on, with fused MMVQ2 and fused SwiGLU enabled on both sides.

- baseline SHA256: `f7254271342a273042f88b21af7267f2fe5a06340ba68a9fc765746090a645aa`
- fused SHA256: `f7254271342a273042f88b21af7267f2fe5a06340ba68a9fc765746090a645aa`
- debug confirmation: `418` `ggml_sycl_try_fuse_rms_norm_mul` calls in a 1-token debug decode

## Results

Single B70, `SYCL2`, 512 prompt / 512 output:

- fused MMVQ2 + SwiGLU baseline: `24.657839 tok/s`
- plus RMS_NORM+MUL fusion: `24.960284 tok/s`
- total throughput: `47.655433 tok/s`

Two B70s, `SYCL2/SYCL1`, tensor split `1/1`, `-ub 128`, 512 prompt / 512 output:

- plus RMS_NORM+MUL fusion: `42.106013 tok/s`
- total throughput: `75.570584 tok/s`
- stddev: `0.011783 tok/s`

Three B70s, `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`, `-ub 128`, 512 prompt / 512 output:

- previous fused MMVQ2 + SwiGLU: `46.804859 tok/s`
- plus RMS_NORM+MUL fusion: `49.366188 tok/s`
- total throughput: `79.667255 tok/s`
- stddev: `0.486931 tok/s`
- LocalMaxxing: `cmoujcois00esld01c5s6bwht`

Four B70 assist split, selector `2,1,3,0`, tensor split `1/1/1/0.05`:

- outcome: failed before benchmark JSON with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during `MUL_MAT`
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/rms-norm-mul-20260506/tensor4-assist005-sg2-p512n512/rmsmul-p512n512-r2-ub32-20260506T204412Z.log`
- decision: do not submit this failed 4x diagnostic to LocalMaxxing

## Interpretation

This is a quality-preserving launch reduction that helps every valid Q4_0 shape tested. The biggest gain is in the 3-B70 tensor split path, where the Q4_0 GGUF result now nearly matches the current static FP8 TP4+n-gram result while keeping the original GGUF target.

The single-B70 result still does not reach the Windows Q4_0 target, so the next software target should be a larger fused projection path, especially fused output-projection plus allreduce/residual epilogues or a broader same-activation GEMV group.
