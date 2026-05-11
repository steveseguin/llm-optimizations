import os

import torch
import torch.distributed as dist

import minimax_ar_fused_rms_xpu  # noqa: F401


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    group_name = "0"
    # PyTorch's functional collective registry uses the default process group
    # name in this standalone torchrun smoke.
    x = torch.full((2, 8), rank + 1, dtype=torch.float16, device=f"xpu:{local_rank}")
    residual = torch.ones_like(x)
    weight = torch.ones((8,), dtype=torch.float16, device=f"xpu:{local_rank}")

    out, residual_out = torch.ops.minimax_ar_fused_rms_xpu.ar_fused_add_rms(
        x, residual, weight, group_name, 1.0e-6
    )
    torch.xpu.synchronize()
    if rank == 0:
        print(
            {
                "out_mean": float(out.float().mean().cpu()),
                "residual_mean": float(residual_out.float().mean().cpu()),
            },
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

