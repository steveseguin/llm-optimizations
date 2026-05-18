# MiniMax Greedy Skip Logits FP32 Negative

Date: 2026-05-18

## Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s
- Baseline recipe: exact MiniMax router-logits path into llm-scaler INT4 MoE work-sharing decode, XPU FlashAttention v2, PIECEWISE graph, `MAX_BATCHED_TOKENS=512`

## Candidate

The candidate added `VLLM_XPU_GREEDY_SKIP_LOGITS_FP32=1` to the vLLM V1 sampler. Under a narrow guard, it skipped the normal `logits.to(torch.float32)` conversion and sampled greedily from the original XPU logits when all of these were true:

- all requests are greedy;
- no logprobs are requested;
- no penalties, allowed-token mask, bad-words filter, or logits processors are active;
- logits are on XPU.

The quality argument was that fp16-to-fp32 conversion cannot change ordering for existing finite fp16 logits when no processors or penalties are applied. The strict gates still had to prove this for the benchmark path.

## Quality

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact: expected token hash `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: expected token hash `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite n64 r2: pass
- arithmetic repeat n64 r16: pass
- extended sixpack n64 r2: pass

Summary JSON:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-greedy-skip-logits-fp32-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T124155Z-summary.json`

## Result

Benchmark shape: p512/n1536, ctx2048, batch 1, TP4, two repeats.

- Repeat 1: `81.237399` output tok/s, `108.316532` total tok/s
- Repeat 2: `81.861443` output tok/s, `109.148591` total tok/s
- Mean: `81.549421` output tok/s, `108.732562` total tok/s

This is slower than the promoted `81.758267` output tok/s mean and essentially within run variance.

## Decision

Do not promote and do not submit to LocalMaxxing. The candidate is quality-clean, but it does not improve decode speed. The active runtime sampler was restored to the promoted baseline after this result was recorded.

Useful learning: the sampler fp32 conversion is not the current bottleneck for the promoted full-logits path. Continue focusing on the lm-head/final-logits boundary, collectives, and MoE/projection epilogue structure.
