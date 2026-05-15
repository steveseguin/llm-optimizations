# MiniMax Q/K Decode SumMean Negative, 2026-05-15

## Purpose

Screen one source-level MiniMax M2.7 AutoRound TP4 decode optimization without
lowering the quality bar. The target path was Q/K RMSNorm because eager timing
still shows it as the largest single local decode bucket.

## Candidate

Temporary default-off patch:

```bash
VLLM_MINIMAX_QK_NORM_DECODE_SUMMEAN=1
VLLM_MINIMAX_QK_NORM_DECODE_SUMMEAN_MAX_TOKENS=1
```

For decode token counts `<= 1`, the candidate changed Q/K variance from
`q.pow(2).mean(dim=-1)` / `k.pow(2).mean(dim=-1)` to
`q.square().sum(dim=-1) / hidden_size` / `k.square().sum(dim=-1) / hidden_size`.
The Q/K variance allreduce, weights, dtype, TP size, graph recipe, and sampler
were otherwise unchanged.

## Result

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-decode-candidates/qk-decode-summean/compiled-piecewise-raw145-qk-decode-summean-ctx2048-n64-20260515T232436Z.json`
- Log:
  `/home/steve/bench-results/minimax-m2.7-decode-candidates/qk-decode-summean/compiled-piecewise-raw145-qk-decode-summean-ctx2048-n64-20260515T232436Z.log`
- Expected raw canary token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed token hash:
  `21404821eb70a2ee3de9e82c039b5cbb5c9eef884c5019579f442c6a272a9c5a`
- Output: `64` generated tokens, `4` distinct token ids, `0` NUL tokens, and
  `0` non-space control characters.
- Text collapsed into repeated `Answer in English.` phrases.

A no-candidate fresh-cache control was then run to separate the source change
from cold compile instability:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-decode-candidates/cold-cache-control/compiled-piecewise-raw145-cold-cache-control-ctx2048-n64-20260515T233904Z.json`
- Cache:
  `/mnt/fast-ai/vllm-cache-exp/minimax-cold-cache-control-20260515T233904Z`
- Result: passed with the expected token hash
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`,
  `28` distinct token ids, `0` NUL tokens, and `0` non-space control
  characters.

## Decision

Reject. This did not produce hard token-0/NUL corruption, but it changed greedy
output and collapsed token diversity. The clean cold-cache control passed, so
this is attributable to the candidate rather than generic compile instability.
It was not benchmarked or submitted to LocalMaxxing. The temporary runtime
patch was removed after the failed canary.

Artifacts:

- `data/minimax-m27-qk-decode-summean-negative-20260515.json`
- `patches/vllm-minimax-qk-decode-summean-negative-20260515.patch`
