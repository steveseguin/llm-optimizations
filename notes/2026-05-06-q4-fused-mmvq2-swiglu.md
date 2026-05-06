# Q4_0 Fused MMVQ2 + SwiGLU

Date: 2026-05-06

## Goal

Reduce decode overhead in Qwen3.6 27B `Q4_0` GGUF without changing weights, quantization, KV dtype, sampling, or GPU power. The target pattern is the FFN gate/up pair: two Q4_0 matvecs share the same activation and immediately feed split SwiGLU.

## Patch

Added an opt-in SYCL path behind `GGML_SYCL_FUSE_MMVQ2_SWIGLU=1`.

- `ggml/src/ggml-sycl/mmvq.hpp`: declared `ggml_sycl_op_mul_mat_vec_q_fused2_swiglu`.
- `ggml/src/ggml-sycl/mmvq.cpp`: added a reordered Q4_0/Q8_1 fused kernel that accumulates both matvecs and writes `silu(gate) * up` directly to the GLU output tensor.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: added the env gate, graph matcher, and safety checks.

The matcher only accepts:

- Q4_0 gate/up weights;
- the same F32 activation input;
- split `GGML_OP_GLU` with `GGML_GLU_OP_SWIGLU`;
- contiguous non-split SYCL buffers;
- no explicit op or matmul stats enabled.

Patch artifact: `patches/llama-cpp-sycl-fused-mmvq2-swiglu-current-20260506.patch.gz.b64`.

## Correctness

Greedy 8-token `llama-completion` stdout matched baseline byte-for-byte:

- baseline SHA256: `a7514e8196ec963459785822b3fcf25b1743096a4bdd5ec746225a7c9a29be19`
- fused SHA256: `a7514e8196ec963459785822b3fcf25b1743096a4bdd5ec746225a7c9a29be19`

## Results

Single B70, 512 prompt / 512 output:

- baseline fused MMVQ2: `24.567164 tok/s`
- fused MMVQ2 + SwiGLU: `24.657839 tok/s`
- delta: `+0.37%`

Three B70 tensor split, `SYCL2/SYCL1/SYCL3`, `-ts 1/1/1`, 512 prompt / 512 output:

- baseline with `-ub 128`: `45.745560 tok/s`
- fused MMVQ2 + SwiGLU with `-ub 128`: `46.804859 tok/s`
- total tok/s: `75.217668`
- LocalMaxxing: `cmougm58m00dpld012rbm9rbs`

## Notes

Default `-ub 512` now fails the 3-B70 512/512 graph reservation with `GGML_ASSERT(buffer) failed` in `ggml_backend_buffer_get_size()` while the meta backend allocates simple compute buffers. `-ub 128` avoids the failure and should be used for 512-context 3-B70 validation until the meta-buffer sizing/null-buffer path is fixed.

The fusion is quality-preserving, but the single-card gain is small. This confirms FFN SwiGLU launch removal is useful but not enough to close the remaining single-B70 Linux gap to the Windows Q4_0 result. The next high-value target remains a broader same-activation GEMV fusion or a fused norm+projection path that reduces more launches and intermediate writes per layer.
