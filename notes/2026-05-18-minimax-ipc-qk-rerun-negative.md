# MiniMax Q/K RMS IPC rerun

Date: 2026-05-18

Scope: rerun the direct IPC prototype for MiniMax M2.7 Q/K RMS variance reduction on 4x Intel Arc Pro B70. The goal was to see whether a custom Level Zero IPC path could replace the tiny XCCL Q/K RMS allreduces in the decode graph.

Command:

```bash
cd /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu_ipc
MINIMAX_QK_IPC_BENCH=1 \
MINIMAX_QK_IPC_VALIDATE=0 \
MINIMAX_QK_IPC_BARRIER=0 \
MINIMAX_QK_IPC_SINGLE_KERNEL=1 \
MINIMAX_QK_IPC_COUNTER=1 \
MINIMAX_QK_IPC_SEQ=0 \
MINIMAX_QK_IPC_ITERS=20 \
MINIMAX_QK_IPC_WARMUP=5 \
MINIMAX_QK_IPC_TOKENS=1 \
MINIMAX_QK_IPC_SLOTS=64 \
MINIMAX_QK_IPC_TIMEOUT_ITERS=100000 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZE_AFFINITY_MASK=0,1,2,3 \
CCL_TOPO_P2P_ACCESS=1 \
PYTHONPATH=. \
/home/steve/.venvs/vllm-xpu/bin/torchrun --standalone --nproc-per-node=4 test_ipc_qk_var.py
```

Result:

- Validation sample still returned expected values: `qk_var=[[26.5, 265.0]]`
- Benchmark shape: 1 token, 20 measured iterations, 5 warmup iterations, 64 IPC slots
- Barrier: disabled
- Single-kernel mode: enabled
- Elapsed: `8.154141 s`
- Mean latency: `407.707070 ms/iter`
- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/qk-ipc-microbench-20260518T030255Z.log`

Decision: reject this direct IPC Q/K variance path for decode. It is orders of magnitude too slow for the per-token Q/K RMS collective, even before integration into the model graph.

Takeaway: keep the Q/K RMS optimization focus inside the existing graph/XCCL path or fuse the variance with surrounding operations. A separate Python-launched IPC collective is not competitive for this tiny reduction.
