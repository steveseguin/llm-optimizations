# 2026-05-04 N-Gram Stall and Q4 Small-F32 Addendum

## FP8 N-Gram Draft Count

Validated best so far remains `num_speculative_tokens=4`:

- `46.066742 tok/s` output on 512 prompt / 512 output.
- LocalMaxxing: `cmorre1hq000fi30421gxpv3j`.

Negative follow-ups:

- `num_speculative_tokens=6` stalled during initialization/profile. Log included `No available shared memory broadcast block found in 60 seconds`; all four workers were CPU-hot.
- `num_speculative_tokens=5` also stalled before useful output. Rank 0 was CPU-hot while the other workers were mostly idle.

Decision: treat 4 draft tokens as the practical ceiling for this vLLM/XPU n-gram path right now. Next tune `prompt_lookup_min/max` around 4 rather than increasing draft count.

## Q4_0 Small-F32 Allreduce

Patch: `GGML_SYCL_COMM_SMALL_F32=1`, env-gated, targeting contiguous F32 20 KB allreduce tensors on 2 or 3 backends.

A/B on Qwen3.6 27B Q4_0 GGUF, 3x B70 selector `2,1,3`, 512 prompt / 128 output:

- Control: `42.773387 tok/s`.
- Small-F32 diagnostic: `34.984690 tok/s`.

Decision: this implementation is not useful. A single 256-work-item kernel underutilizes the B70 despite the small tensor size. If continuing the tiny-reduction route, keep full-range parallelism and only test lower-overhead dependency barriers, or move to fused MMVQ/DMMV allreduce epilogue work.
