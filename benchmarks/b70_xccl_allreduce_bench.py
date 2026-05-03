import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    sizes = [
        4 * 1024,
        64 * 1024,
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    ]

    if rank == 0:
        print("bytes, dtype, iters, avg_ms, payload_GBps")

    for nbytes in sizes:
        numel = nbytes // 2
        x = torch.ones(numel, dtype=torch.float16, device=f"xpu:{local_rank}")
        iters = 200 if nbytes <= 64 * 1024 else 50 if nbytes <= 16 * 1024 * 1024 else 20

        for _ in range(10):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.xpu.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start

        avg = elapsed / iters
        if rank == 0:
            gbps = nbytes / avg / 1e9
            print(f"{nbytes}, fp16, {iters}, {avg * 1000:.3f}, {gbps:.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
