# MiniMax Q/K RMS XPU IPC Prototype

Prototype for the cross-process peer-memory part of a future XPU equivalent of
vLLM's CUDA `minimax_allreduce_rms_qk` path.

This is also wired into a default-off vLLM experiment, but the standalone
harness is still the cleanest correctness test:

1. each TP-style process allocates a small XPU mailbox tensor;
2. each process exports its mailbox with Level Zero `zeMemGetIpcHandle`;
3. all processes exchange handles through `torch.distributed`;
4. each process opens every peer mailbox with `zeMemOpenIpcHandle`;
5. a SYCL kernel writes local Q/K variance scalars to its mailbox and polls peer
   mailboxes to compute the average.

Build with the oneAPI 2025.3 compiler to match the active PyTorch XPU runtime:

```bash
cd /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu_ipc
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
/home/steve/.venvs/vllm-xpu/bin/python setup.py build_ext --inplace
```

Run:

```bash
cd /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu_ipc
MINIMAX_QK_IPC_SINGLE_KERNEL=1 \
MINIMAX_QK_IPC_COUNTER=1 \
MINIMAX_QK_IPC_SEQ=0 \
MINIMAX_QK_IPC_ITERS=50 \
MINIMAX_QK_IPC_TOKENS=32 \
MINIMAX_QK_IPC_SLOTS=3 \
MINIMAX_QK_IPC_TIMEOUT_ITERS=100000 \
PYTHONPATH=. /home/steve/.venvs/vllm-xpu/bin/torchrun --standalone \
  --nproc-per-node=4 test_ipc_qk_var.py
```

The scalar-sequence variant can be tested with `MINIMAX_QK_IPC_SCALAR_SEQ=1`
instead of `MINIMAX_QK_IPC_COUNTER=1`. It passes this standalone harness, but
the vLLM integration was much too slow, so the device-counter path remains the
primary fusion candidate.

Benchmark mode:

```bash
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
PYTHONPATH=. /home/steve/.venvs/vllm-xpu/bin/torchrun --standalone \
  --nproc-per-node=4 test_ipc_qk_var.py
```

On 2026-05-10 this measured about `416 ms/iter` for a one-token `[1, 2]`
payload, far slower than XCCL's decode-sized tiny allreduce microbench. Also
avoid slot wrap in no-barrier tests; if a rank misses a sequence and a peer
overwrites that slot, the equality-based poll can hang.
