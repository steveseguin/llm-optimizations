import os
import time

import torch
import torch.distributed as dist


def parse_ints(env_name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def iters_for_tokens(tokens: int) -> int:
    override = os.environ.get("B70_AGRS_ITERS")
    if override:
        return int(override)
    if tokens <= 4:
        return 500
    if tokens <= 128:
        return 200
    return 50


def synchronize() -> None:
    torch.xpu.synchronize()
    dist.barrier()


def time_region(iters: int, fn) -> float:
    for _ in range(8):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    return (time.perf_counter() - start) / iters


def all_gather_list(x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    outs = [
        torch.empty((size, x.shape[1]), dtype=x.dtype, device=x.device)
        for size in sizes
    ]
    dist.all_gather(outs, x)
    return torch.cat(outs, dim=0)


def all_gather_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    out = torch.empty(
        (x.shape[0] * world_size, x.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(out, x)
    return out


def all_gather_vllm_xpu_equal(x: torch.Tensor, world_size: int) -> torch.Tensor:
    # Mirrors the equal-size branch in XpuCommunicator.all_gatherv as of the
    # 2026-05-14 local vLLM tree.
    out = torch.empty(
        (x.shape[0] * world_size, x.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather([out], x)
    return out


def all_gather_padded_uneven(x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    max_size = max(sizes)
    if x.shape[0] == max_size:
        padded = x.contiguous()
    else:
        padded = torch.empty((max_size, x.shape[1]), dtype=x.dtype, device=x.device)
        padded[: x.shape[0]].copy_(x)
        padded[x.shape[0] :].zero_()
    gathered = torch.empty(
        (max_size * len(sizes), x.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, padded)
    return torch.cat(
        [
            gathered[rank * max_size : rank * max_size + size]
            for rank, size in enumerate(sizes)
        ],
        dim=0,
    )


def reduce_scatter_list(x: torch.Tensor, sizes: list[int], rank: int) -> torch.Tensor:
    chunks = list(x.split(sizes, dim=0))
    out = torch.empty((sizes[rank], x.shape[1]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter(out, chunks)
    return out


def reduce_scatter_tensor(x: torch.Tensor, tokens: int, hidden: int) -> torch.Tensor:
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x)
    return out


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    hidden = int(os.environ.get("B70_AGRS_HIDDEN", "6144"))
    token_sizes = parse_ints("B70_AGRS_TOKEN_SIZES", [1, 4, 32, 128, 512])

    if rank == 0:
        ccl_env = {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("CCL_")
        }
        print(f"world_size={world_size} hidden={hidden} ccl_env={ccl_env}")
        print("case,tokens,hidden,iters,avg_ms,payload_mib")

    for tokens in token_sizes:
        x = torch.ones((tokens, hidden), dtype=torch.float16, device=f"xpu:{local_rank}")
        gathered = torch.ones(
            (tokens * world_size, hidden),
            dtype=torch.float16,
            device=f"xpu:{local_rank}",
        )
        iters = iters_for_tokens(tokens)

        cases = [
            ("all_gather_into_tensor_equal", lambda: all_gather_tensor(x, world_size)),
            (
                "all_gather_list_equal",
                lambda: all_gather_list(x, [tokens] * world_size),
            ),
            (
                "reduce_scatter_tensor_equal",
                lambda: reduce_scatter_tensor(gathered, tokens, hidden),
            ),
            (
                "reduce_scatter_list_equal",
                lambda: reduce_scatter_list(
                    gathered, [tokens] * world_size, rank
                ),
            ),
        ]
        if os.environ.get("B70_AGRS_INCLUDE_VLLM_COMPAT", "0") == "1":
            cases.insert(
                2,
                (
                    "all_gather_vllm_xpu_equal",
                    lambda: all_gather_vllm_xpu_equal(x, world_size),
                ),
                )

        if os.environ.get("B70_AGRS_INCLUDE_UNEVEN", "0") == "1":
            sizes = [tokens + offset for offset in range(world_size)]
            uneven_x = torch.ones(
                (sizes[rank], hidden),
                dtype=torch.float16,
                device=f"xpu:{local_rank}",
            )
            uneven_gathered = torch.ones(
                (sum(sizes), hidden),
                dtype=torch.float16,
                device=f"xpu:{local_rank}",
            )
            uneven_cases = [
                (
                    "all_gather_padded_uneven",
                    lambda: all_gather_padded_uneven(uneven_x, sizes),
                ),
                (
                    "all_gather_list_uneven",
                    lambda: all_gather_list(uneven_x, sizes),
                ),
                (
                    "reduce_scatter_list_uneven",
                    lambda: reduce_scatter_list(uneven_gathered, sizes, rank),
                ),
            ]
        else:
            uneven_cases = []

        for case, fn in cases:
            avg = time_region(iters, fn)
            if rank == 0:
                payload_mib = tokens * hidden * torch.float16.itemsize / 1024 / 1024
                print(
                    f"{case},{tokens},{hidden},{iters},"
                    f"{avg * 1000:.4f},{payload_mib:.4f}",
                    flush=True,
                )

        for case, fn in uneven_cases:
            avg = time_region(iters, fn)
            if rank == 0:
                payload_mib = sum(sizes) * hidden * torch.float16.itemsize / 1024 / 1024
                print(
                    f"{case},{tokens},{hidden},{iters},"
                    f"{avg * 1000:.4f},{payload_mib:.4f}",
                    flush=True,
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
