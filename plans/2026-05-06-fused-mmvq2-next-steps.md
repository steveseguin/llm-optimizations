# 2026-05-06 Q4 tensor-split next steps

## Current best Qwen3.6 27B Q4_0 GGUF result

- Model: `unsloth/Qwen3.6-27B-GGUF`, Q4_0 GGUF.
- Hardware: 3x Intel Arc Pro B70 32GB, selector order `level_zero:2,1,3`.
- Engine: llama.cpp SYCL/Level Zero build with Q8 cache, scheduler async tensor copy disabled, peer async copy enabled, single-kernel allreduce, event barrier, post-allreduce sync, meta fused allreduce-add, and gated adjacent Q4_0 MMVQ2 fusion.
- Result: `45.954130 tok/s` decode, `118.362712 tok/s` prompt, `66.202667 tok/s` computed total at 512 prompt / 512 generate / 3 repeats.
- Quality: full-logit deterministic repeat passed for 16 greedy decode steps.
- LocalMaxxing: compact required-field submission succeeded after the API recovered. ID `cmotmnnm6000aqu01uzb9wk12`. Detailed payload with notes/flags returned `500 Internal Server Error`, so the public row currently lacks full launch metadata.
- Benchmark files:
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-triple213-p512n512-r3-20260506T051928Z.jsonl`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quality-cleared-singlekernel-syncafter-fusemmvq2-triple213-p512n512-r3-20260506T051928Z.log`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/syncafter-long-and-4x-20260506T051503Z`

## What changed

- Added an opt-in `GGML_SYCL_FUSE_MMVQ2=1` path that fuses adjacent Q4_0 reordered matvecs sharing the same one-token activation.
- Added a token/logit correctness harness that hashes full logits at each greedy decode step and avoids the noisy multi-GPU PPL oracle.
- Added diagnostic collective controls:
  - `GGML_SYCL_COMM_SYNC_AFTER=1`, which made 3x single-kernel allreduce repeatable.
  - `GGML_SYCL_COMM_SYNC_READY=1`, which did not fix 3x by itself.
  - `GGML_SYCL_COMM_STAGED_ROOT_COPY=1`, which is experimental and failed repeatability.
- The first fused MMVQ2 targets are `ffn_gate + ffn_up` pairs and `Vcur + Kcur` pairs.

## Quality status

- `GGML_SYCL_FUSE_MMVQ2=1` is quality-cleared on the fast 2x and 3x paths using the token/logit harness.
- The best 2x path is `40.933579 tok/s` decode and passed a 16-step full-logit repeat.
- The best 3x path is `45.954130 tok/s` decode and passed a 16-step full-logit repeat.
- 4x with sync-after is repeatable in a 4-step full-logit smoke test but only reaches `34.920977 tok/s`, so it is not a throughput win.
- Multi-GPU PPL remains a noisy oracle and should not be used for pass/fail quality on these tensor-split paths.

## Correctness findings

- `GGML_SYCL_ASYNC_CPY_TENSOR=1` is unsafe on tensor split: it diverged on 2x even with custom allreduce disabled.
- `GGML_SYCL_ASYNC_PEER_COPY=1` is stable in isolation and can remain enabled.
- Generic custom allreduce remains unsafe due in-place peer-copy/add races.
- 2x single-kernel custom allreduce is stable with scheduler async tensor copy disabled.
- 3x/4x single-kernel custom allreduce needs `GGML_SYCL_COMM_SYNC_AFTER=1`; ready-sync alone did not fix 3x, after-sync did.
- Experimental staged-root allreduce failed repeatability on 2x/3x/4x and should stay off.

## Next steps

1. Retry a compact-but-detailed LocalMaxxing payload after the five-minute POST window, or ask for an edit endpoint, so the public row gets command flags/context/notes.
2. Clean up `GGML_SYCL_COMM_SYNC_AFTER=1` from a diagnostic wait-all-streams into a narrower correctness fence, ideally only on peer destinations that consume root-written allreduce output.
3. Gate or rewrite the generic custom allreduce path so it cannot silently use the in-place race.
4. Keep `GGML_SYCL_ASYNC_CPY_TENSOR=0` in all recommended tensor-split recipes until the scheduler copy API can propagate a correct destination event.
5. Port the exact Q4_0/Q8_1 ESIMD fused2 prototype into llama.cpp behind a second opt-in gate, then re-run single, 2x, 3x, and 4x benchmarks.
6. Investigate deeper FFN fusion around `ffn_gate + ffn_up + swiglu`, because it may remove another graph node and reduce memory traffic.
7. Continue 4x work by reducing collective overhead; current 4x is repeatable but slower than 3x.
8. Keep FP8/vLLM TP4 and Q4_0 GGUF tracks separate in notes and submissions, because they trade different quantization and runtime behavior.
