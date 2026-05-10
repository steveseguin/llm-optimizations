# MiniMax Q/K RMS XPU IPC Prototype

Prototype for the cross-process peer-memory part of a future XPU equivalent of
vLLM's CUDA `minimax_allreduce_rms_qk` path.

This is not wired into vLLM yet. The first target is a standalone correctness
test:

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
PYTHONPATH=. /home/steve/.venvs/vllm-xpu/bin/torchrun --standalone \
  --nproc-per-node=4 test_ipc_qk_var.py
```

