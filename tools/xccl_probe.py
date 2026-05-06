import os
import sys

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    mode = sys.argv[1] if len(sys.argv) > 1 else "init"

    torch.xpu.set_device(local_rank)
    dist.init_process_group("xccl")
    print(f"rank {rank} init ok", flush=True)

    if mode in {"barrier", "allreduce"}:
        dist.barrier(device_ids=[local_rank])
        print(f"rank {rank} barrier ok", flush=True)

    if mode == "allreduce":
        x = torch.ones(1, dtype=torch.float16, device=f"xpu:{local_rank}")
        dist.all_reduce(x)
        torch.xpu.synchronize()
        print(f"rank {rank} allreduce ok {float(x.cpu()[0])}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
