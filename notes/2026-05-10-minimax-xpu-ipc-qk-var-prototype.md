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
PYTHONPATH=. /home/steve/.venvs/vllm-xpu/bin/torchrun --standalone \
  --nproc-per-node=4 test_ipc_qk_var.py
```

Output:

```text
rank=0 qk_var=[[2.5, 25.0], ...] ok=True
rank=1 qk_var=[[2.5, 25.0], ...] ok=True
rank=2 qk_var=[[2.5, 25.0], ...] ok=True
rank=3 qk_var=[[2.5, 25.0], ...] ok=True
```

## What Worked

- Exporting a PyTorch XPU tensor allocation with `zeMemGetIpcHandle`.
- Sharing handles through `torch.distributed` using the `xccl` backend.
- Opening peer mailbox tensors with `zeMemOpenIpcHandle` in each TP worker.
- Packing Level Zero pointers into a device `int64` pointer table after
  two's-complement conversion for addresses above signed-int64 range.
- Reading peer mailbox memory from a SYCL kernel and producing the exact global
  average for the synthetic `[tokens, 2]` MiniMax Q/K variance payload.

## What Did Not Work Yet

The first single-kernel attempt wrote the local mailbox and then spin-polled
peer mailboxes. It was not reliable: late-launching rank 3 usually saw all
peers, while earlier ranks read stale sentinel values from one or more peers.
Adding a host `dist.barrier()` before launch was not enough because the race is
inside cross-device kernel ordering/visibility, not process launch order alone.

The passing path is two-phase:

1. write local mailbox with a normal XPU copy;
2. `torch.xpu.synchronize()`;
3. `dist.barrier()`;
4. run the peer-read reduction kernel.

This is probably not fast enough for vLLM decode as-is, but it confirms the
hard foundation: cross-process peer mailbox reads from SYCL work on the B70
stack.

## Next Work

The next prototype should replace the naive sentinel spin with a real
device-side sequence protocol:

- per-rank mailbox slots include sequence counters separate from payload;
- use SYCL atomics or Level Zero memory scopes where supported for local writes;
- avoid cross-card remote atomics because Level Zero reports only `ACCESS` for
  cross-card pairs;
- consider a two-kernel graph-safe version if single-kernel ordering remains
  unreliable, then measure whether it beats oneCCL for the tiny `[tokens, 2]`
  MiniMax Q/K variance payload.

Only after this peer-memory allreduce is correct should it be fused with RMS
apply in the MiniMax attention path.

