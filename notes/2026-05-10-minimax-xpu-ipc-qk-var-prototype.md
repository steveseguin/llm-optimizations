# MiniMax XPU IPC Q/K Variance Prototype

## Summary

I added a standalone `minimax_qk_rms_xpu_ipc` SYCL/PyTorch extension that uses
Level Zero IPC handles from real XPU tensors. It proves that four TP-style
processes can exchange peer mailbox pointers and run an XPU kernel that reads
all peer mailboxes to compute the global Q/K variance average.

This is not wired into vLLM yet. It is a correctness stepping stone for a
future XPU equivalent of vLLM's CUDA `minimax_allreduce_rms_qk` fusion.

## Result

Build:

```bash
cd /home/steve/llm-optimizations-publish/experiments/minimax_qk_rms_xpu_ipc
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
CC=/opt/intel/oneapi/compiler/2025.3/bin/icx \
/home/steve/.venvs/vllm-xpu/bin/python setup.py build_ext --inplace
```

Run:

```bash
MINIMAX_QK_IPC_SINGLE_KERNEL=1 \
MINIMAX_QK_IPC_SEQ=1 \
MINIMAX_QK_IPC_ITERS=50 \
MINIMAX_QK_IPC_TOKENS=32 \
MINIMAX_QK_IPC_TIMEOUT_ITERS=100000 \
PYTHONPATH=. /home/steve/.venvs/vllm-xpu/bin/torchrun --standalone \
  --nproc-per-node=4 test_ipc_qk_var.py
```

Output:

```text
rank=0 iter=49 qk_var=[[51.5, 515.0], ...] ok=True
rank=1 iter=49 qk_var=[[51.5, 515.0], ...] ok=True
rank=2 iter=49 qk_var=[[51.5, 515.0], ...] ok=True
rank=3 iter=49 qk_var=[[51.5, 515.0], ...] ok=True
```

## What Worked

- Exporting a PyTorch XPU tensor allocation with `zeMemGetIpcHandle`.
- Sharing handles through `torch.distributed` using the `xccl` backend.
- Opening peer mailbox tensors with `zeMemOpenIpcHandle` in each TP worker.
- Packing Level Zero pointers into a device `int64` pointer table after
  two's-complement conversion for addresses above signed-int64 range.
- Reading peer mailbox memory from a SYCL kernel and producing the exact global
  average for the synthetic `[tokens, 2]` MiniMax Q/K variance payload.
- A single-kernel sequence-counter variant now passes 4-rank stress:
  50 iterations, 32 token rows, local write, system fence, sequence publish,
  remote sequence polling, and peer payload reduction inside one SYCL kernel.

## What Did Not Work Yet

The first single-kernel attempt used `-0.0` float sentinels in the payload
mailbox itself. It was not reliable: late-launching rank 3 usually saw all
peers, while earlier ranks read stale sentinel values from one or more peers.
Adding a host `dist.barrier()` before launch was not enough because the race is
inside cross-device kernel ordering/visibility, not process launch order alone.

The first sequence-counter stress used the old 500M spin timeout and looked
hung. A shorter `MINIMAX_QK_IPC_TIMEOUT_ITERS=100000` completed correctly and
is a better test harness default for failure isolation. The high timeout may
still be useful in a real integration, but it hides liveness failures during
experiments.

The original passing path was two-phase:

1. write local mailbox with a normal XPU copy;
2. `torch.xpu.synchronize()`;
3. `dist.barrier()`;
4. run the peer-read reduction kernel.

That two-phase version is probably not fast enough for vLLM decode as-is, but
it confirmed the hard foundation: cross-process peer mailbox reads from SYCL
work on the B70 stack.

## Next Work

The next prototype should move the sequence-counter path toward vLLM:

- wrap mailbox allocation, Level Zero handle exchange, peer pointer table
  construction, slot assignment, and sequence generation in a Python manager;
- integrate behind a default-off MiniMax env flag replacing
  `tensor_model_parallel_all_reduce(qk_var) / tp_world_size`;
- validate p1/n8 logits against the default oneCCL path before any throughput
  test;
- avoid cross-card remote atomics because Level Zero reports only `ACCESS` for
  cross-card pairs;
- keep the two-phase variant as a fallback if compiled graph capture or launch
  ordering rejects the single-kernel protocol.

Only after this peer-memory allreduce is correct should it be fused with RMS
apply in the MiniMax attention path.
