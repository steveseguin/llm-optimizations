# MiniMax M2.7 AutoRound BF16 U4 Larger-Context Result

## Result

The 4x B70 vLLM/XPU TP4 path with the llm-scaler u4 decode bridge clears the
30 tok/s target at a larger prompt window:

| Label | Prompt | Output | Max model len | Output tok/s | Total tok/s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `vllm-minimax-m27-autoround-bf16-u4-tp4-p1024-n256` | 1024 | 256 | 2048 | 31.833 | 159.166 | success |
| `vllm-minimax-m27-autoround-bf16-u4-tp4-p1536-n256` | 1536 | 256 | 2048 | n/a | n/a | failed: no KV memory |

Successful run:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1024n256-20260509T233854Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1024n256-20260509T233854Z.json
```

The successful run reported:

```text
Throughput: 0.12 requests/s, 159.17 total tokens/s, 31.83 output tokens/s
GPU KV cache size: 2,688 tokens
Maximum concurrency for 2,048 tokens per request: 1.31x
Available KV cache memory: 0.16 GiB
Model loading took 28.96 GiB memory and 348.60 seconds
```

## Interpretation

This is a valid quality-preserving MiniMax result: same AutoRound INT4 W4A16
target weights, BF16 hidden state, vLLM/XPU TP4, FlashAttention2, no speculative
decode, no expert dropping, no sampling change, and no power-limit change.

The result is also a capacity warning. At `max_model_len=2048`, the model has
only about `0.16 GiB` of KV headroom per card. The p1536/n256 attempt failed
after profiling with:

```text
Available KV cache memory: -0.3 GiB
ValueError: No available memory for the cache blocks
```

So the next MiniMax bottleneck is not only decode kernel speed. Larger context
requires freeing GPU memory before the run can even allocate KV blocks.

## Next Work

- Treat n-gram and suffix speculative decode as workload-specific, lossless
  options. They are worth testing on repetitive prompts, but random-token
  leaderboard runs are unlikely to benefit.
- Keep DFlash marked blocked for this setup until the MiniMax drafter can
  allocate KV reliably; current attempts either run out of KV or lose the XPU
  device.
- Look for memory-headroom wins: compile range, graph/cache overhead, MoE config
  memory, and any avoidable FP32/BF16 temporary allocations around the W4A16
  MoE path.
