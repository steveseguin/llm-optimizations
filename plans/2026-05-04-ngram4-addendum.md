# 2026-05-04 N-Gram4 FP8 Addendum

Status: new best static FP8 path.

## Result

`vrfai/Qwen3.6-27B-FP8`, vLLM XPU/FA2, TP4, auto/BF16 KV, language-model-only, 4x Intel Arc Pro B70.

- `num_speculative_tokens=4`
- `prompt_lookup_min=2`
- `prompt_lookup_max=5`
- 512 prompt / 512 output
- 1 warmup + 3 measured iterations
- `avg_latency=11.114308836 s`
- `tokSOut=46.066742`
- `tokSTotal=92.133484`
- LocalMaxxing: `cmorre1hq000fi30421gxpv3j`

This is ahead of the prior TP4 FP8 FA2 baseline (`41.503 tok/s`) and the current Q4_0 TP3 validation (`~41.659 tok/s`).

## Quality Notes

This preserves the static FP8 target model and auto/BF16 KV. N-gram speculative decode verifies draft tokens with the target model, so expected output distribution is target-model quality, modulo vLLM's warning that `min_p` and `logit_bias` are unavailable under speculative decoding. No power-limit changes.

## Next

- Sweep `num_speculative_tokens=6` and `8` on the same 512/512 shape.
- If one wins, validate with 3-5 measured iterations.
- Then tune prompt lookup window.
- Keep MTP separate; its current TP4 issue is startup/XCCL/draft synchronization, not a proven decode speed path yet.
