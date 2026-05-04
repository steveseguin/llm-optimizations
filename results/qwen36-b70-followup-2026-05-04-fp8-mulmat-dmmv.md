# Qwen3.6 27B B70 Follow-Up: FP8 Artifact Check, Q4_0 MUL_MAT Profiling, DMMV Experiments

Date: 2026-05-04
Host: Ubuntu 24.04, Intel Arc Pro B70 32GB GPUs
Model under primary test: `Qwen3.6-27B-Q4_0.gguf`

## FP8 Artifact State

The actual FP8 model is already downloaded locally at:

```text
/home/steve/models/qwen3.6-27b-fp8-hf
```

This is `Qwen/Qwen3.6-27B-FP8` in HF/Safetensors format, not GGUF. Its config reports:

```json
{
  "quant_method": "fp8",
  "fmt": "e4m3",
  "activation_scheme": "dynamic",
  "weight_block_size": [128, 128]
}
```

Hugging Face search found repos named like `Qwen3.6-27B-FP8-Q4_K_M-GGUF`, but their actual downloadable GGUF is `qwen3.6-27b-fp8-q4_k_m.gguf`, meaning a Q4_K_M GGUF derived from an FP8 source, not native FP8 GGUF. `RedHatAI/Qwen3.6-27B-FP8` has the same HF/Safetensors file shape, so it was not duplicated locally.

## New Diagnostic Patch

Added env-gated profiling in `ggml_sycl_op_mul_mat`:

```text
GGML_SYCL_MUL_MAT_STATS=1
```

It splits explicit-sync timing into stages such as:

- `quantize_main`
- `quantize_chunk`
- `peer_copy_q8`
- `mul_mat_kernel`
- `dst_copy`
- `split_wait`

This mode intentionally inserts waits and is diagnostic only. Absolute timings are perturbed, but stage counts and relative launch structure are useful.

## Single-Card MUL_MAT Findings

Single B70 selector 2, `-p 0 -n 1`:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mulmat-profile-single2-p0n1-20260504T145852Z.*
```

Results:

```text
quantize_main: 345 calls, 284.483 ms explicit-sync total
mul_mat_kernel: 497 calls, 98.455 ms explicit-sync total
```

Single B70 selector 2, `-p 512 -n 1`:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mulmat-profile-single2-p512n1-20260504T145928Z.*
```

Decode delta after prompt was roughly:

```text
quantize_main: ~345 calls, ~255 ms explicit-sync total
mul_mat_kernel: ~497 calls, ~46 ms explicit-sync total
```

Interpretation: normal throughput is much faster than these explicit-sync totals, but the launch structure is clear. Q4_0 decode pays hundreds of activation Q8_1 quantization launches per token before the reordered MMVQ kernels.

## DMMV Retest

Global DMMV no longer crashed on a one-token smoke in the rebuilt tree, but it remains slower for throughput.

Forced DMMV, single B70, 512 prompt / 64 output:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-dmmv-retake-single2-p512n64-20260504T150247Z.jsonl
15.518 tok/s decode
```

Same-shape MMVQ control:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mmvq-control-single2-p512n64-20260504T150332Z.jsonl
22.683 tok/s decode
```

Decision: do not globally prefer DMMV.

## Selective `result_output` DMMV Experiment

Added another env-gated diagnostic switch:

```text
GGML_SYCL_DMMV_OUTPUT=1
```

This uses DMMV only for the final `result_output` matmul and leaves other Q4_0 matmuls on reordered MMVQ.

Short no-warmup run:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-dmmv-output-single2-p512n64-20260504T151433Z.jsonl
20.995 tok/s decode
```

Warm 256-token validation:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-dmmv-output-validate-single2-p512n256-20260504T151544Z.jsonl
24.007 tok/s decode
```

Same-build MMVQ control:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mmvq-control2-single2-p512n256-20260504T151637Z.jsonl
24.426 tok/s decode
```

Decision: selective output-projection DMMV is also a real-throughput loss. Keep the switch diagnostic-only.

## Next Work

The next plausible Q4_0 single-card target is not DMMV. It is reducing launch overhead or improving graph scheduling around the current reordered MMVQ path:

- replace explicit-sync timing with event/device timestamp profiling where possible;
- inspect SYCL graph update/replay behavior around repeated quantize plus matvec dispatches;
- prototype batching or fusing Q8_1 activation quantization only if it avoids excessive repeated quantization work;
- keep DMMV switches disabled for benchmark submissions.

No power-limit or overclocking changes were made.
