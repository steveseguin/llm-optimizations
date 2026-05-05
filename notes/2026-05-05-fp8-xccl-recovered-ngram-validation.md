# Qwen3.6 FP8 XCCL Recovery and N-Gram Validation

Date: 2026-05-05

## Summary

Standalone XCCL recovered without a reboot, so I resumed static FP8 TP4 validation on the `vrfai/Qwen3.6-27B-FP8` checkpoint.

The earlier 2-iteration `50.193 tok/s` lookup `2/4` screen did not fully reproduce as a 3-iteration validation, but the recovered/default CCL environment produced a new validated FP8 best:

- output throughput: `47.674832 tok/s`;
- total throughput: `95.349664 tok/s`;
- LocalMaxxing: `cmos3pnqo000kkz04o4aiup22`.

## XCCL Gate

Command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
CCL_ZE_IPC_EXCHANGE=sockets \
python -m torch.distributed.run --standalone --nproc_per_node=2 /home/steve/b70_xccl_allreduce_bench.py
```

Result:

- completed allreduce rows from 4 KiB through 256 MiB;
- 64 MiB: `41.53 GB/s`;
- 256 MiB: `41.80 GB/s`;
- log: `/home/steve/bench-results/qwen36-fp8-vllm/xccl-standalone-2rank-post-q4-localwrite-20260505T035142Z.log`.

## Validated Run

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- engine: patched vLLM/XPU `0.20.1`;
- GPUs: 4x Intel Arc Pro B70;
- TP/PP: `4/1`;
- quantization: `compressed-tensors` FP8;
- attention: XPU FlashAttention2;
- KV cache: `auto`;
- context length: `1024`;
- prompt/output: `512/512`;
- speculative decode: n-gram, `num_speculative_tokens=4`, lookup min/max `2/4`;
- CCL: `CCL_ATL_TRANSPORT=ofi`, default IPC/topology recognition.

Results:

- latencies: `10.523862531`, `11.585105772`, `10.109288830` seconds;
- average latency: `10.739419044330134` seconds;
- output throughput: `47.674832 tok/s`;
- total throughput: `95.349664 tok/s`;
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T035653Z.json`;
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T035653Z.log`.

## Adjacent Negative Screens

- same lookup `2/4` config with forced `CCL_ZE_IPC_EXCHANGE=sockets` and `CCL_TOPO_P2P_ACCESS=1`: `43.342333 tok/s`;
- lookup max `5` under default IPC/topology: `42.159878 tok/s`.

Decision: current validated FP8 best is lookup min/max `2/4` with default IPC/topology recognition.
