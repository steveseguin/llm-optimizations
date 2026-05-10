# MiniMax CCL P2P-Off Negative

## Summary

Tried disabling oneCCL topology P2P access:

```text
CCL_P2P=0
USE_LLM_SCALER_MOE=1
XPU_GRAPH=0
INPUT_LEN=512 OUTPUT_LEN=256 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4
```

The run loaded the model and reached warmup, then hung during or just after
request dispatch. It was stopped manually instead of waiting for the full
timeout.

Log:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T001921Z.log
```

## Observations

The model load looked normal:

```text
Loading weights took 348.12 seconds
Model loading took 28.96 GiB memory and 349.663168 seconds
Initial profiling/warmup run took 1.28 s
```

After that, vLLM repeatedly reported a stuck broadcast:

```text
No available shared memory broadcast block found in 60 seconds.
```

## Interpretation

`CCL_TOPO_P2P_ACCESS=0` is not a useful workaround for B70 TP=4. It appears to
break or severely stall the multi-process dispatch/collective path for this
workload. Keep the default P2P topology path enabled.

This result is not suitable for LocalMaxxing because no inference throughput was
produced.
