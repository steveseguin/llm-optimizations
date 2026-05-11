# External Suggestion Triage, 2026-05-11

The newest suggested MiniMax recipe overlaps with several local screens. This
file records what is already known versus what remains worth a controlled
retest.

## Already Tested Or Known Negative Here

- `TP4+EP4`: tested and slower than pure TP4 for batch-1 decode. The best
  recorded EP-tuned long run was around `30.91` output tok/s, versus the
  quality-cleared pure TP4 anchor at `37.552538` output tok/s and
  `50.070051` total tok/s.
- `--enforce-eager`: tested as a fallback path and clearly slower. A p64/n128
  MiniMax screen dropped from `33.745` output tok/s compiled to `18.007`
  output tok/s eager.
- n-gram speculation: tested. Plain ngram and `ngram_gpu` were negative or
  stalled on this XPU stack. The useful future path is target-verified
  EAGLE/DFlash/MTP-style drafting, not generic ngram on random prompts.
- FP8 KV cache: tested for MiniMax TP4. It increased reported KV capacity but
  the p512/n512 serving run completed zero requests and died in `sample_tokens`
  after shared-memory broadcast timeouts.
- full Arctic Inference plugin: not a clean install target here because Arctic
  0.1.2's vLLM extra pins `vllm==0.11.0` and the broader plugin path is
  CUDA-oriented. A suffix-only subset is now runnable; see
  `notes/2026-05-11-minimax-arctic-suffix-smoke.md`.
- power-limit changes: intentionally out of scope unless the user changes the
  target. The current focus is software/runtime optimization, not raising card
  power.

## Worth A Controlled Retest Later

- `VLLM_XPU_ENABLE_XPU_GRAPH=1`: keep as a graph-cache/scheduler retest, but
  isolate `VLLM_CACHE_ROOT` and record cold versus warm behavior. Prior graph
  behavior on this stack produced misleading KV-cache artifacts.
- `fuse_minimax_qk_norm`: vLLM source has a MiniMax-specific Q/K norm fusion
  pass, but the upstream implementation expects non-XPU custom ops. Local helper
  and direct-call XPU experiments were negative. A clean retest is still useful
  only if it ports the boundary into a real XPU kernel or compiler lowering.
- `kv-cache-dtype fp8_e5m2` with `--block-size 128`: keep as a longer-context
  capacity/debug screen. It should not be promoted as a decode improvement until
  `sample_tokens` stability is fixed.
- `--disable-custom-all-reduce`: current vLLM XPU logs already show
  `disable_custom_all_reduce=True` by default. Re-enabling a custom collective
  should be measured, not assumed.
- throughput sweeps from 1 to 128 sequences: useful for capacity reporting and
  LocalMaxxing total tok/s context, but secondary to the single-session decode
  target.
- AR-W4A16 versus UD-IQ4_XS quality/latency side-by-side: useful for explaining
  why the AutoRound path is the primary four-B70 target. Keep this separate
  from quality-preserving optimization of the full AutoRound model.

## Current Priority After Triage

Stay on the MiniMax AutoRound TP4 quality path and keep the next optimization
work focused on communication boundaries:

1. Q/K RMS variance allreduce plus adjacent reshape/RoPE/RMS work.
2. Hidden-state allreduce followed by residual add/RMSNorm.
3. MoE/output-projection epilogues where collective waits force graph breaks.
4. Speculative decoding only after target verification and acceptance behavior
   can be measured on XPU.

