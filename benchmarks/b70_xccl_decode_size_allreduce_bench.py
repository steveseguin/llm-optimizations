import os
import time

import torch
import torch.distributed as dist


def bench(dtype: torch.dtype, numel: int, iters: int) -> tuple[float, float]:
    local_rank = int(os.environ["LOCAL_RANK"])
    x = torch.ones(numel, dtype=dtype, device=f"xpu:{local_rank}")

    for _ in range(20):
        dist.all_reduce(x)
    torch.xpu.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.xpu.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start
    avg_s = elapsed / iters
    nbytes = x.numel() * x.element_size()
    return avg_s * 1000.0, nbytes / avg_s / 1e9


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    # MiniMax M2.7 hidden_size is 3072. Batch-1 tensor-parallel decode
    # reductions are expected to be in this small-payload regime.
    cases = [
        ("minimax_hidden_fp16", torch.float16, 3072, 1000),
        ("minimax_hidden_fp32", torch.float32, 3072, 1000),
        ("two_hidden_fp16", torch.float16, 6144, 1000),
        ("two_hidden_fp32", torch.float32, 6144, 1000),
        ("small_20kb_fp32", torch.float32, 5120, 1000),
        ("small_64kb_fp16", torch.float16, 32768, 1000),
    ]

    if rank == 0:
        print("label,dtype,numel,bytes,iters,avg_ms,payload_GBps")

    for label, dtype, numel, iters in cases:
        avg_ms, gbps = bench(dtype, numel, iters)
        if rank == 0:
            nbytes = numel * torch.empty((), dtype=dtype).element_size()
            print(
                f"{label},{str(dtype).replace('torch.', '')},"
                f"{numel},{nbytes},{iters},{avg_ms:.6f},{gbps:.3f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
