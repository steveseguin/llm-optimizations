# MiniMax Current-Recipe Extra Screens

Date: 2026-05-13

These screens stayed inside the validated MiniMax M2.7 AutoRound current-best
recipe unless listed as a delta:

- TP4, FP16 activations
- XPU graph with graph partitioning
- attention delayed allreduce enabled
- `--block-size 256`
- `MAX_BATCHED_TOKENS=512`
- `--no-enable-prefix-caching`
- current AOT cache: `/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-20260513T171301Z`
- AOT hash: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

## Results

| Run | Delta | Prompt/output | Total tok/s | Output tok/s | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| MiniMax logits MoE | `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1` | 512/1536 | `97.619374` | `73.214530` | negative near-baseline |
| stream interval retry | `--stream-interval 16` | 512/1536 | no result | no result | stopped after repeated shared-memory waits; already known negative |
| disable hybrid KV | `--disable-hybrid-kv-cache-manager` | 512/1536 | `97.159428` | `72.869571` | negative/no-op |

## Decisions

- Keep `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS` unset. The exact logits
  path is quality-preserving, but it did not beat the current repeat mean.
- Keep the default hybrid KV decision. MiniMax showed the same 17,920-token KV
  cache size with `--disable-hybrid-kv-cache-manager`, and decode regressed.
- Do not revisit `--stream-interval` on this recipe unless another scheduler
  change makes output-processing overhead visible again. It was already measured
  negative earlier and this retry stalled before graph capture.

None of these results were submitted to LocalMaxxing because they are not new
promoted results.
