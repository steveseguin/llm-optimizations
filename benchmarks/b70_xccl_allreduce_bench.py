import os
import time

import torch
import torch.distributed as dist


def parse_modes() -> list[str]:
    raw = os.environ.get("B70_ALLREDUCE_MODES")
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return ["inplace", "clone", "empty_copy"]


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


def run_allreduce(mode: str, x: torch.Tensor) -> torch.Tensor:
    if mode == "inplace":
        out = x
    elif mode == "clone":
        # vLLM's XPU communicator clones while torch.compile is tracing,
        # because the all-reduce must appear out-of-place to Dynamo.
        out = x.clone()
    elif mode == "empty_copy":
        out = torch.empty_like(x)
        out.copy_(x)
    else:
        raise ValueError(f"unknown B70_ALLREDUCE_MODES entry: {mode}")
    dist.all_reduce(out)
    return out


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    sizes = parse_sizes()
    modes = parse_modes()

    if rank == 0:
        ccl_env = {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("CCL_")
        }
        print(f"world_size={world_size} ccl_env={ccl_env}")
        print("mode, bytes, dtype, iters, avg_ms, payload_GBps")

    for mode in modes:
        for nbytes in sizes:
            numel = nbytes // 2
            x = torch.ones(numel, dtype=torch.float16, device=f"xpu:{local_rank}")
            iters = iters_for_size(nbytes)

            for _ in range(10):
                run_allreduce(mode, x)
            torch.xpu.synchronize()
            dist.barrier()

            start = time.perf_counter()
            for _ in range(iters):
                run_allreduce(mode, x)
            torch.xpu.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start

            avg = elapsed / iters
            if rank == 0:
                gbps = nbytes / avg / 1e9
                print(
                    f"{mode}, {nbytes}, fp16, {iters}, "
                    f"{avg * 1000:.3f}, {gbps:.2f}"
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
