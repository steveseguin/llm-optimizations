# MiniMax M2.7 AutoRound N-Gram Speculative Decode

## Summary

N-gram speculative decoding is currently a negative path for MiniMax M2.7
AutoRound on the 4x B70 vLLM/XPU setup.

| Label | Prompt | Output | Spec config | Status | Output tok/s | Total tok/s |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `vllm-minimax-m27-ngram4-p512-n256` | 512 | 256 | ngram, 4 tokens, lookup 2-4 | failed: no KV | n/a | n/a |
| `vllm-minimax-m27-ngram2-gpu095-p512-n256` | 512 | 256 | ngram, 2 tokens, lookup 2-4, `gpu_memory_utilization=0.95` | success, negative | 12.267 | 36.800 |

## What Happened

The first attempt used:

```text
--speculative-config {"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":4}
```

It failed during KV allocation:

```text
Available KV cache memory: -0.3 GiB
ValueError: No available memory for the cache blocks
```

The second attempt reduced speculative depth and increased the vLLM memory cap:

```text
--gpu-memory-utilization 0.95
--speculative-config {"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":2,"prompt_lookup_max":4}
```

It completed:

```text
Throughput: 0.05 requests/s, 36.80 total tokens/s, 12.27 output tokens/s
Available KV cache memory: 0.65 GiB
GPU KV cache size: 10,944 tokens
```

## Interpretation

This does not reduce target model quality: vLLM verifies speculative tokens.
The problem is performance and capacity. For this random-token benchmark,
n-gram speculation has poor match characteristics, disables async scheduling in
vLLM, adds speculative capture/KV pressure, and is much slower than the
non-speculative BF16 u4 path.

Keep n-gram and suffix speculation on the list only for repetitive real
workloads such as code edits, agent loops, or prompts with explicit repeated
context. Do not use n-gram for the current random MiniMax leaderboard-style
benchmark unless a workload-specific acceptance test shows a large accepted-token
rate.
