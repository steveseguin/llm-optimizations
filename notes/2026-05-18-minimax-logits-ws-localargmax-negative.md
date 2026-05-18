# MiniMax M2.7 Logits-WS Local-Argmax Follow-Up

Date: 2026-05-18

## Result

`VLLM_XPU_LOCAL_ARGMAX_DECODE=1` plus `VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1` was tested on top of the current promoted logits-to-work-sharing MiniMax path:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- default XPU FlashAttention v2
- XPU graph, PIECEWISE compile
- p512/n1536, ctx2048, batch 1, TP4

The candidate passed the strict quality gate but was slower than the promoted full-logits path:

- Run 1: `72.260669` output tok/s, `96.347558` total tok/s
- Run 2: `73.700102` output tok/s, `98.266803` total tok/s
- Mean: `72.980385` output tok/s, `97.307181` total tok/s

Current promoted baseline remains `81.758267` output tok/s and `109.011023` total tok/s.

## Quality

All quality gates passed:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: pass
- arithmetic repeat: 16/16 deterministic exact `42`
- extended sixpack: pass

This means local argmax is quality-safe for this guarded greedy/no-logprobs/no-processors benchmark shape, but not useful for speed on top of the current logits-WS recipe.

## Decision

Do not promote and do not submit to LocalMaxxing. The result is a quality-passed negative datapoint, not a leaderboard improvement.

The main takeaway is that the current bottleneck is not the full-vocab logits gather/sampler path after the logits-to-work-sharing MoE change. Keep focusing on decode collectives and GPU-side fusion around Q/K variance RMS, attention delayed residual allreduce, MoE output allreduce, projection/lm-head boundaries, and prefill efficiency.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-localargmax-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T145921Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-localargmax-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T145921Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T151502Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T151754Z.json`
- Repro data: `data/minimax-m27-logits-ws-localargmax-negative-20260518.json`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.
