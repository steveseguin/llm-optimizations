# MiniMax Long-Output Validation

## Result

Ran the current quality-preserving vLLM/XPU MiniMax AutoRound path with a longer
generation window:

```text
USE_LLM_SCALER_MOE=1 CCL_IPC=default XPU_GRAPH=0 DTYPE=float16
INPUT_LEN=512 OUTPUT_LEN=1024 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4
```

The run completed:

```text
Throughput: 0.04 requests/s, 53.90 total tokens/s, 35.93 output tokens/s
Available KV cache memory: 1.02 GiB
GPU KV cache size: 17,216 tokens
```

Log/json:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1024-20260510T003153Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1024-20260510T003153Z.json
```

## Interpretation

This is a valid longer-generation result and remains comfortably above the
initial 30 tok/s goal, but it does not beat the p512/n512 high of `37.136187`
output tok/s.

That matters: the current performance ceiling is not just prefill amortization.
As the active context grows, per-token decode cost rises enough that the longer
run drops to `35.933290` output tok/s.

Keep p512/n512 as the current best speed point. Use p512/n1024 as a more
realistic longer-generation validation datapoint.
