# MiniMax CCL Worker Affinity Negative, 2026-05-10

## Summary

oneCCL worker affinity did not improve MiniMax M2.7 AutoRound INT4 TP4 throughput on the four B70 setup. Both runs completed and reused the same AOT graph cache, so this is not another XCCL initialization hang or cold-cache KV artifact. It is simply not a useful speed knob in the current stack.

## Setup

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: vLLM/XPU TP4, dtype `float16`, INC/AutoRound WNA16 quantization
- Decode path: `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- Prompt/output: p512/n512
- Context: `max_model_len=2048`
- GPUs: 4x Intel Arc Pro B70 32 GB
- Power/frequency: unchanged
- AOT cache: `4799a3c8468de261861723fba07480ef61e010f504245a62e5e93f4e9aef8e22`
- KV cache: 17,216 GPU tokens in both runs

## Results

| Setting | Total tok/s | Output tok/s | Notes |
| --- | ---: | ---: | --- |
| `CCL_WORKER_AFFINITY=auto` | 72.992583 | 36.496292 | Completed; no speed win |
| `CCL_WORKER_AFFINITY=0,1,2,3` | 71.135368 | 35.567684 | Completed; slower |

Reference points for comparison:

- Accepted p512/n512 speed point: 39.610585 output tok/s.
- Accepted p512/n1024 speed point: 40.303730 output tok/s.
- Conservative p512/n1536 quality anchor: 37.552538 output tok/s.

## Interpretation

This reinforces the earlier oneCCL conclusion: raw XCCL is not the obvious bottleneck. The local XCCL microbenchmark measured the relevant tiny MiniMax allreduces in the tens of microseconds, while runtime throughput is dominated by graph boundaries, attention/KV/projection scheduling, Q/K RMS placement, and MoE epilogue shape.

Keep default oneCCL worker settings. Do not submit these runs to LocalMaxxing because they are not improvements.

## Logs

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ccl-affinity/vllm-minimax-m27-autoround-tp4-p512n512-20260510T144526Z.log`
- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ccl-affinity/vllm-minimax-m27-autoround-tp4-p512n512-20260510T144804Z.log`
