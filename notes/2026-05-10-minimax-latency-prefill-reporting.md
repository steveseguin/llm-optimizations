# MiniMax Latency And Prefill Reporting, 2026-05-10

## Result

Added `scripts/bench-vllm-minimax-autoround-latency-xpu.sh` for single-request
latency probes of the MiniMax AutoRound INT4 path.

Clean TP4 run:

| Shape | Avg latency | p50 latency | p90 latency | Total request tok/s | Output-equivalent tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/n128 | `3.665656 s` | `3.662841 s` | `3.673611 s` | `174.594` | `34.919` |

Artifacts:

- log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-latency-p512n128/vllm-minimax-m27-autoround-latency-tp4-p512n128-20260510T223048Z.log`
- JSON:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-latency-p512n128/vllm-minimax-m27-autoround-latency-tp4-p512n128-20260510T223048Z.json`
- structured summary:
  `data/minimax-m27-autoround-latency-p512n128-20260510.json`

## Reporting Lesson

This benchmark is useful for user-facing single-request latency and an
end-to-end output-equivalent rate, but it does not expose TTFT, prefill tok/s,
or steady decode tok/s as separate fields.

For LocalMaxxing submissions, do not invent `ttftMs` or `tokSPrefill` from this
result. Continue reporting `tokSTotal` and `tokSOut` from `vllm bench
throughput`, and add true TTFT/prefill only after one of these is available:

- a vLLM serving trace with request-level TTFT and token timing;
- a small instrumented local benchmark that records prefill completion and
  decode-token intervals;
- a vLLM upstream benchmark mode that emits those fields for XPU.

## Operational Note

The run loaded the model with only a few GiB of free system RAM. It completed,
but `/usr/bin/time -v` reported `816261` major page faults and about `245 GB` of
filesystem input. That is not the steady-state decode ceiling, but it makes
full p512/n1536 validation loops expensive. Keep screens short and only promote
variants that show a clear short-run signal.

## Target Update

For the MiniMax AutoRound INT4 path, `30-40` output tok/s is now bring-up
history. The current quality-cleared p512/n1536 anchor is `37.552538` output
tok/s and `50.070051` total tok/s, so the active targets are:

- `60+` output tok/s non-speculative, quality-preserving;
- `75+` output tok/s with verified speculative decoding as the first stretch;
- higher stretch targets only if acceptance rate and target verification are
  recorded beside the performance result.
