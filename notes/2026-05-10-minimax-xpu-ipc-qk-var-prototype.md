# MiniMax XPU IPC Q/K Variance Prototype

## Summary

I added a standalone `minimax_qk_rms_xpu_ipc` SYCL/PyTorch extension that uses
Level Zero IPC handles from real XPU tensors. It proves that four TP-style
processes can exchange peer mailbox pointers and run an XPU kernel that reads
all peer mailboxes to compute the global Q/K variance average.

This started as a standalone correctness stepping stone for a future XPU
equivalent of vLLM's CUDA `minimax_allreduce_rms_qk` fusion. A default-off
vLLM hook now exists, but the standalone tests remain the cleanest place to
separate mailbox protocol correctness from vLLM scheduler and graph behavior.

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
MINIMAX_QK_IPC_COUNTER=1 \
MINIMAX_QK_IPC_SEQ=0 \
MINIMAX_QK_IPC_ITERS=50 \
MINIMAX_QK_IPC_TOKENS=32 \
MINIMAX_QK_IPC_SLOTS=3 \
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
- A device-counter variant also passes 4-rank stress while reusing only three
  mailbox slots across 50 iterations. This is the more relevant compiled-graph
  candidate because the expected sequence is read from device memory instead
  of passed as a Python integer that TorchDynamo could capture as a constant.
- The same device-counter path also passed a prefill-sized payload smoke:
  512 token rows, 10 iterations, and three reused mailbox slots, ending with
  exact `[11.5, 115.0]` rows on all ranks.
- A scalar-sequence variant also passes the standalone stress cases. It
  publishes one sequence value per mailbox slot instead of one sequence value
  per payload element. Standalone checks passed at 50 iterations / 32 rows and
  at 10 iterations / 512 rows, both with three reused slots.

## What Did Not Work Yet

The first single-kernel attempt used `-0.0` float sentinels in the payload
mailbox itself. It was not reliable: late-launching rank 3 usually saw all
peers, while earlier ranks read stale sentinel values from one or more peers.
Adding a host `dist.barrier()` before launch was not enough because the race is
inside cross-device kernel ordering/visibility, not process launch order alone.

The first host-sequence-counter stress used the old 500M spin timeout and looked
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

The scalar-sequence path is correct in the standalone harness, but it is not a
vLLM speed path. When wired into MiniMax TP4, a p1/n4 eager smoke completed at
only about `0.03` output tok/s and the compiled opt-in path completed at about
`0.02` output tok/s after a long compile/warmup. Keep the scalar variant as a
negative artifact unless the protocol is moved into a lower-level fused kernel.

## Next Work

The next prototype should move the sequence-counter path toward vLLM:

- wrap mailbox allocation, Level Zero handle exchange, peer pointer table
  construction, slot assignment, and sequence generation in a Python manager;
- prefer the device-counter op for graph experiments so sequence values are not
  Python constants;
- do not prioritize the scalar-sequence op in vLLM; it is useful as a simpler
  correctness reference but was far too slow in the real MiniMax path;
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
