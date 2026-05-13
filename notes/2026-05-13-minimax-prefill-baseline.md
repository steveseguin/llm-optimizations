# MiniMax Long-Context Prefill Baseline, 2026-05-13

## Purpose

Screen the prefill/long-context path without changing the current MiniMax decode
recipe. This uses the same quality-preserving TP4 setup as the current best
p512/n1536 run, but with a longer prompt shape:

- input tokens: 4096
- output tokens: 512
- `MAX_MODEL_LEN=8192`
- `MAX_BATCHED_TOKENS=512`
- `--block-size 256`
- `--no-enable-prefix-caching`
- XPU graph plus graph partition
- llm-scaler INT4 MoE decode path
- MiniMax attention delayed allreduce

## Results

The first p4096/n512 run had to create/load a long-context graph path and should
be treated as a cold diagnostic:

| Run | Class | Output tok/s | Total tok/s |
| --- | --- | ---: | ---: |
| 1 | cold long-context graph | 38.403 | 345.636 |
| 2 | warm long-context graph | 58.208 | 523.874 |
| 3 | warm long-context graph | 58.893 | 530.033 |

Warm summary:

- warm output tok/s mean: `58.550`
- warm output tok/s min/max: `58.208` / `58.893`
- warm total tok/s mean: `526.954`
- warm total tok/s min/max: `523.874` / `530.033`

## Interpretation

The long-context path is viable under the current decode recipe. The cold run
shows why prefill numbers need a warmed graph condition; after the graph exists,
the p4096/n512 total throughput is around `524-530` tok/s and the 512-token
decode tail is around `58-59` tok/s.

This should not be promoted as a new decode optimization. It is a useful
prefill/long-context baseline. Any prefill-specific setting change, such as a
larger `MAX_BATCHED_TOKENS`, should rerun the p512/n1536 decode repeatability
gate before becoming a default.

## Artifacts

- Data summary:
  `data/minimax-m27-prefill-baseline-p4096n512-20260513.json`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T215513Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T220053Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T220317Z.json`
