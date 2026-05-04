# 2026-05-04 N-Gram, MTP, and Q4 Allreduce Addendum

## FP8 N-Gram

Status: promoted from correctness smoke to valid optimization track.

- TP4 static FP8, 512 prompt / 256 output: `42.245 tok/s` output.
- TP4 static FP8, 512 prompt / 512 output: `42.489 tok/s` output.
- Prior TP4 FP8 FA2 512/512 baseline: `41.503 tok/s`.
- LocalMaxxing validation result: `cmorr43b30004jj04h4hhb6v1`.

Next work:

- Sweep `num_speculative_tokens` on the 512/512 shape.
- Sweep `prompt_lookup_min/max` after the token count sweep.
- Keep auto/BF16 KV; FP8 KV was slower and has quality risk.

## MTP

Status: still blocked before useful generation.

Observed failure: TP4 MTP resolves target `Qwen3_5ForConditionalGeneration` and draft `Qwen3_5MTP`, initializes XCCL ranks, then hangs. `strace` showed a worker spinning in `sched_yield`.

Read-only review points to these likely boundaries:

- `vllm/v1/worker/xpu_worker.py`, `XPUWorker.init_device` distributed init and XPU all-reduce warmup.
- `vllm/v1/worker/gpu_model_runner.py`, speculative setup and drafter creation.
- `vllm/v1/spec_decode/draft_model.py`, draft TP handling.
- `vllm/model_executor/models/qwen3_5_mtp.py`, `Qwen3_5MultiTokenPredictor` TP collectives.
- `vllm/v1/worker/xpu_model_runner.py`, CUDA compatibility wrapper for XPU events/streams.

Next diagnostics:

- TP1 MTP compatibility smoke.
- `--enforce-eager`.
- MTP JSON `enforce_eager`.
- `disable_padded_drafter_batch`.

Do not run heavy MTP benchmarks until the startup path is proven.

## Q4_0 Allreduce

Status: next implementation target is narrower than a full fused matmul epilogue.

Current trace shape: 128 reductions per token, each `20480` bytes, contiguous F32 direct `MUL_MAT` outputs.

Recommended first patch boundary:

- Add an env-gated small-F32 allreduce fast path in `ggml_backend_sycl_comm_allreduce_tensor()` for `nbytes == 20480`, `type == GGML_TYPE_F32`, and `n_backends in {2,3}`.
- Keep meta/model graph semantics unchanged.
- Use it to quantify the remaining tiny-reduction overhead floor before larger fused MMVQ/allreduce work.

Larger follow-up if the fast path helps:

- Pass peer output pointers/events into selected MMVQ/DMMV epilogues for `linear_attn_out`, `attn_output`, and `ffn_out` partial matmul outputs.
- This crosses `ggml_sycl_op_mul_mat()`, `ggml_sycl_op_mul_mat_vec_q()`, and meta backend scheduling, so it is the second patch, not the first.
