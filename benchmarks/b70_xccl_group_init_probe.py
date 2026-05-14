import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank: int, message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"{now} rank={rank} {message}", flush=True)


def timed(rank: int, label: str, fn) -> float:
    dist.barrier()
    start = time.perf_counter()
    log(rank, f"{label} start")
    result = fn()
    torch.xpu.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start
    log(rank, f"{label} done elapsed_s={elapsed:.6f}")
    return elapsed, result


def allreduce_group(rank: int, label: str, group) -> None:
    x = torch.ones(8, dtype=torch.float16, device=f"xpu:{rank}")
    timed(rank, f"{label} xpu_allreduce", lambda: dist.all_reduce(x, group=group))


def cpu_node_check(rank: int, label: str, group) -> None:
    x = torch.zeros(dist.get_world_size(group=group), dtype=torch.int32, device="cpu")
    x[dist.get_rank(group=group)] = 1
    timed(rank, f"{label} cpu_allreduce", lambda: dist.all_reduce(x, group=group))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(local_rank)

    ccl_env = {key: os.environ[key] for key in sorted(os.environ) if key.startswith("CCL_")}
    if rank == 0:
        print(f"world_size={world_size} ccl_env={ccl_env}", flush=True)

    log(rank, "init_process_group start")
    init_kwargs = {"backend": "xccl"}
    if os.environ.get("B70_GROUP_PROBE_DEVICE_ID", "0") == "1":
        init_kwargs["device_id"] = torch.device(f"xpu:{local_rank}")
    dist.init_process_group(**init_kwargs)
    log(rank, "init_process_group done")
    allreduce_group(local_rank, "default_pg", dist.group.WORLD)

    group_count = int(os.environ.get("B70_GROUP_PROBE_COUNT", "8"))
    do_gloo = os.environ.get("B70_GROUP_PROBE_GLOO", "1") == "1"
    ranks = list(range(world_size))
    xpu_groups = []
    gloo_groups = []
    for idx in range(group_count):
        label = f"group{idx}"
        log(rank, f"{label} new_xccl start")
        xpu_group = dist.new_group(ranks, backend="xccl")
        xpu_groups.append(xpu_group)
        log(rank, f"{label} new_xccl done")
        allreduce_group(local_rank, label, xpu_group)

        if do_gloo:
            log(rank, f"{label} new_gloo start")
            gloo_group = dist.new_group(ranks, backend="gloo")
            gloo_groups.append(gloo_group)
            log(rank, f"{label} new_gloo done")
            cpu_node_check(local_rank, label, gloo_group)

    if rank == 0:
        print("probe_complete", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"probe_failed: {exc!r}", file=sys.stderr, flush=True)
        raise
