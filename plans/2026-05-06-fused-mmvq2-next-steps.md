# 2026-05-06 fused MMVQ2 next steps

## Current best Qwen3.6 27B Q4_0 GGUF result

- Model: `Qwen/Qwen3.6-27B`, Q4_0 GGUF.
- Hardware: 3x Intel Arc Pro B70 32GB, selector order `level_zero:2,1,3`.
- Engine: llama.cpp SYCL/Level Zero build with Q8 cache, async copy, single-kernel allreduce, event barrier, meta fused allreduce-add, and gated adjacent Q4_0 MMVQ2 fusion.
- Result: `46.117650 tok/s` decode, `135.810565 tok/s` prompt, `68.854236 tok/s` computed total at 512 prompt / 512 generate / 3 repeats.
- LocalMaxxing id: `cmotg6zz50004jo04eyqocxli`.
- Benchmark files:
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-fusemmvq2-fused-triple213-p512n512-r3-20260506T023132Z.jsonl`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-fusemmvq2-fused-triple213-p512n512-r3-20260506T023132Z.log`

## What changed

- Added an opt-in `GGML_SYCL_FUSE_MMVQ2=1` path that fuses adjacent Q4_0 reordered matvecs sharing the same one-token activation.
- The first fused targets are `ffn_gate + ffn_up` pairs and `Vcur + Kcur` pairs. Qwen3.6 27B exposes 80 such model-level adjacent pairs per token.
- The gate is conservative: it skips split tensors, non-Q4_0 weights, non-F32 activations/results, non-contiguous inputs, and debug/stat modes that need per-op accounting.
- A standalone ESIMD probe now uses exact GGUF Q4_0 and Q8_1 semantics and shows fused2 kernels can save roughly 25-39% versus two separate launches for representative shapes.

## Quality status

- Single-card PPL matched exactly with `GGML_SYCL_FUSE_MMVQ2=0` vs `1`: `PPL = 2.0500 +/- 0.56981` for both.
- 3-card forced non-conversation deterministic generation matched byte-for-byte with temp 0.
- 3-card PPL is not a stable quality oracle yet: a fusion-disabled repeat run varied from `PPL = 2.1587` to `2.3188`.
- Treat `GGML_SYCL_FUSE_MMVQ2=1` as experimental for multi-GPU until a better token/logit correctness harness is built.

## Next steps

1. Build a token/logit correctness harness that avoids the current multi-GPU PPL nondeterminism.
2. Keep `GGML_SYCL_FUSE_MMVQ2=1` opt-in only until that harness clears the 3-card and 4-card paths.
3. Port the exact Q4_0/Q8_1 ESIMD fused2 prototype into llama.cpp behind a second opt-in gate, then re-run single, 3x, and 4x benchmarks.
4. Investigate a deeper FFN fusion around `ffn_gate + ffn_up + swiglu`, because it may remove another graph node and reduce memory traffic.
5. Re-test 4x B70 after the local-kernel changes. Current 4x remains below 3x, so broad root-order sweeps are lower priority than reducing per-device launch/collective overhead.
6. Keep FP8/vLLM TP4 and Q4_0 GGUF tracks separate in notes and submissions, because they trade different quantization and runtime behavior.
