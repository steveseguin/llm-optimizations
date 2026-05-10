# MiniMax B70 MoE Config Retest

## Result

Retested the existing hybrid B70 MoE config with the current BF16 llm-scaler u4
decode path:

```text
VLLM_TUNED_CONFIG_FOLDER=/home/steve/bench-results/minimax-m2.7-autoround-vllm/moe-config-hybrid-m1-default-prefill
USE_LLM_SCALER_MOE=1
INPUT_LEN=512 OUTPUT_LEN=256 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 TP=4
```

The run completed:

```text
Throughput: 0.13 requests/s, 99.79 total tokens/s, 33.26 output tokens/s
Available KV cache memory: 0.16 GiB
GPU KV cache size: 2,688 tokens
```

Log/json:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T000747Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T000747Z.json
```

## Interpretation

The config was definitely used:

```text
Using configuration from .../moe-config-hybrid-m1-default-prefill/E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json for MoE layer.
```

This removes the default-config warning, but it does not improve the current
BF16 llm-scaler u4 p512/n256 path. The stronger non-tuned p512/n256 datapoints
remain in the mid-30 tok/s range.

Conclusion: keep this config archived for older pre-bridge or prefill-specific
experiments, but do not enable it by default for current MiniMax BF16 u4 decode
work.
