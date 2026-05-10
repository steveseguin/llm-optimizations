# MiniMax oneCCL Worker and Built-in Fusion Screens, 2026-05-10

## Goal

Screen low-risk communication knobs after the MiniMax allreduce-boundary
experiments.

Intel oneCCL documents `CCL_WORKER_COUNT` as the number of oneCCL worker threads
and `CCL_WORKER_AFFINITY` as the CPU affinity control for those workers:

- https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-15/environment-variables.html
- https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-14/execution-of-communication-operations.html

vLLM also has a `fuse_allreduce_rms` compilation pass, which is conceptually
close to the fusion we want for MiniMax.

## Findings

### `CCL_WORKER_COUNT=2`

Command shape:

```bash
CCL_WORKER_COUNT=2 \
USE_LLM_SCALER_MOE=1 CCL_P2P=1 XPU_GRAPH=0 \
TP=4 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 \
INPUT_LEN=512 OUTPUT_LEN=512 NUM_PROMPTS=1 DTYPE=float16 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

The run hung during XCCL initialization, before model weight loading. The log
showed oneCCL accepting the setting:

```text
value of CCL_WORKER_COUNT changed to be 2 (default:1)
```

but it did not progress past the early local-rank/ATL setup messages. The run
was killed. No JSON throughput result was produced.

Decision: keep the default oneCCL worker count for vLLM TP4 on this B70 stack.

### vLLM `fuse_allreduce_rms`

The built-in vLLM pass is not usable for this path without new XPU work:

- `vllm/platforms/xpu.py` explicitly disables `fuse_allreduce_rms` on XPU.
- `allreduce_rms_fusion.py` is FlashInfer/ROCm-oriented; it expects FlashInfer
  fused allreduce workspaces or ROCm AITER fused kernels, not Level Zero/XCCL.

Decision: do not try to force-enable the stock pass for MiniMax/XPU. The useful
direction remains an XPU-specific fused allreduce plus residual/RMSNorm path.

## Outcome

No LocalMaxxing submission. Both items are diagnostic negatives:

- `CCL_WORKER_COUNT=2` hangs at XCCL init.
- Built-in allreduce-RMS fusion is intentionally disabled on XPU and lacks the
  required Level Zero backend.
