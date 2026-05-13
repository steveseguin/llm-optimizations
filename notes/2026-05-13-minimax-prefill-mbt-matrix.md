# MiniMax Prefill MBT Matrix, 2026-05-13

## Purpose

Test whether larger prefill chunks improve the p4096/n512 long-context path
without changing the quality-sensitive decode recipe. The fixed recipe stayed:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM `0.20.1-local` XPU
- TP4, FP16
- XPU graph with graph partition
- llm-scaler INT4 MoE
- MiniMax attention delayed allreduce
- `--block-size 256`
- `--no-enable-prefix-caching`
- `MAX_NUM_SEQS=1`

Only `MAX_BATCHED_TOKENS` changed.

## Results

Baseline, previously recorded with `MAX_BATCHED_TOKENS=512`:

| MBT | Warm runs | Output tok/s mean | Total tok/s mean |
| ---: | ---: | ---: | ---: |
| 512 | 2 | 58.550 | 526.954 |

New matrix:

| MBT | Warm runs | Output tok/s mean | Total tok/s mean | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1024 | 2 | 56.159 | 505.428 | negative |
| 2048 | 2 | 55.853 | 502.678 | negative |
| 4096 | 0 | n/a | n/a | interrupted before measurement |

Cold graph-creation runs were also slower:

| MBT | Cold output tok/s | Cold total tok/s |
| ---: | ---: | ---: |
| 1024 | 37.431 | 336.883 |
| 2048 | 36.786 | 331.077 |

## Interpretation

Larger `MAX_BATCHED_TOKENS` did not help this p4096/n512 shape. Both 1024 and
2048 reduced warmed throughput versus the 512 baseline. They also showed long
initialization stalls after AOT load, with repeated shared-memory broadcast wait
messages before profiling/warmup completed. That makes them worse operationally
even before considering the lower measured throughput.

Keep `MAX_BATCHED_TOKENS=512` for the current long-context baseline. No
LocalMaxxing submission was made because this is a negative screen.

## Artifacts

- Data summary:
  `data/minimax-m27-prefill-mbt-matrix-20260513.json`
- Matrix runner:
  `scripts/run-minimax-prefill-mbt-matrix.sh`
- Run manifest:
  `/home/steve/bench-results/minimax-m2.7-prefill-matrix/minimax-prefill-mbt-matrix-20260513T224920Z.runs.jsonl`
- Selected logs/JSONs:
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T225818Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T230628Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T232156Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260513T233004Z.json`

## Next

Return to source-level decode work. The remaining high-value target is still
hidden-state collective fusion or scheduling around attention output projection
and MoE hidden-state allreduces. Prefill chunk sizing does not look like the
next path to a >60 tok/s quality-preserving decode gain.
