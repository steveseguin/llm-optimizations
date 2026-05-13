# MiniMax Async Runtime Screens, 2026-05-13

These screens followed the `compile_sizes=[1]` static decode graph win for
MiniMax M2.7 AutoRound W4A16 on four Arc Pro B70 GPUs. The goal was to check
whether host/runtime scheduling could improve single-request decode without
changing model quality.

## Results

| Test | Prompt/Output | Output tok/s | Total tok/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| reboot baseline, static decode graph | 512/512 | 44.994 | 89.987 | healthy baseline |
| `--stream-interval 32` | 512/512 | 44.508 | 89.016 | negative |
| `--max-num-seqs 2` | 512/512 | 32.618 | 65.235 | negative, new compile shape |
| `--async-engine` | 512/512 | 45.648 | 91.295 | small positive |
| `--async-engine` | 512/1536 | 47.743 | 63.658 | repeat candidate |
| `--async-engine` repeat | 512/1536 | 48.093 | 64.124 | new accepted best |
| `--async-engine --no-enable-prefix-caching` | 512/512 | 46.192 | 92.383 | short-run positive |
| same long repeat | 512/1536 | 47.050 | 62.733 | negative versus accepted best |

`--no-enable-prefix-caching --no-enable-chunked-prefill` did not load under the
normal `max_model_len=2048` / `max_num_batched_tokens=1024` recipe. vLLM warned
that MiniMax does not officially support disabling chunked prefill, then failed
validation because no-chunk mode requires `max_num_batched_tokens >=
max_model_len`. This is not worth pursuing for the quality-preserving speed path.

Disabling prefix caching alone is valid with chunked prefill left enabled, and
it reused the same static decode graph with `16,832` GPU KV-cache tokens. It is
not a promoted speed path: the short p512/n512 run edged above the async-engine
screen, but the p512/n1536 repeat landed below the accepted `48.092807` output
tok/s anchor. Keep prefix caching enabled unless a later source patch changes
the scheduling tradeoff.

## Current Best

The new quality-preserving p512/n1536 anchor is:

- `48.092807` output tok/s and `64.123742` total tok/s.
- TP4, FP16 activations, AutoRound INT4 W4A16 weights.
- `--async-engine`
- `--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1]}'`
- AOT `3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`.
- No speculative decode, no expert dropping, no reduced router precision, no
  power-limit changes.
- LocalMaxxing: `cmp3cgooj0019s401d7p1ks3e`.

This is a small runtime-side gain, not a route to the `60 tok/s` target. The
main path remains source-level execution work: communication/epilogue fusion,
attention/KV scheduling, and MoE bridge/kernel overhead.

## Logs

- reboot baseline: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-reboot-baseline-staticcompile-p512n512-20260513T003008Z`
- stream interval: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-streaminterval32-staticcompile-p512n512-20260513T003242Z`
- no prefix/no chunk validation failure: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-noprefix-nochunk-staticcompile-p512n512-20260513T003532Z`
- max seqs 2: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-maxseq2-staticcompile-p512n512-20260513T003617Z`
- async p512/n512: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-asyncengine-staticcompile-p512n512-20260513T004155Z`
- async p512/n1536: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-asyncengine-staticcompile-p512n1536-20260513T004424Z`
- async p512/n1536 repeat: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-asyncengine-staticcompile-repeat-p512n1536-20260513T004725Z`
- no-prefix p512/n512: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-noprefix-staticcompile-p512n512-20260513T014259Z`
- no-prefix p512/n1536: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-noprefix-staticcompile-p512n1536-20260513T014535Z`
