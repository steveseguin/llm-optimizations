import os
import time

import torch
import torch.distributed as dist


def parse_sizes() -> list[int]:
    raw = os.environ.get("B70_ALLREDUCE_SIZES")
    if raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [
        8,  # MiniMax Q/K variance allreduce, 2 x fp32.
        64,
        1024,
        4096,
        6144,  # MiniMax FP16 hidden allreduce, 3072 x fp16.
        8192,
        64 * 1024,
        1 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    ]


def iters_for_size(nbytes: int) -> int:
    override = os.environ.get("B70_ALLREDUCE_ITERS")
    if override:
        return int(override)
    if nbytes <= 8192:
        return 1000
    if nbytes <= 64 * 1024:
        return 300
    if nbytes <= 16 * 1024 * 1024:
        return 50
    return 20


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    sizes = parse_sizes()

    if rank == 0:
        ccl_env = {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("CCL_")
        }
        print(f"world_size={world_size} ccl_env={ccl_env}")
        print("bytes, dtype, iters, avg_ms, payload_GBps")

    for nbytes in sizes:
        numel = nbytes // 2
        x = torch.ones(numel, dtype=torch.float16, device=f"xpu:{local_rank}")
        iters = iters_for_size(nbytes)

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
