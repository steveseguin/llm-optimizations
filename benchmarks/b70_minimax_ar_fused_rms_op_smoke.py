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
    dtype = torch.float16
    hidden_size = 2048
    torch.manual_seed(20260511 + rank)
    x = (
        torch.randn((3, hidden_size), dtype=dtype, device=f"xpu:{local_rank}") * 0.1
        + rank
    )
    residual = (
        torch.randn((3, hidden_size), dtype=dtype, device=f"xpu:{local_rank}") * 0.1
    )
    weight = torch.randn((hidden_size,), dtype=dtype, device=f"xpu:{local_rank}") * 0.1

    out, residual_out = torch.ops.minimax_ar_fused_rms_xpu.ar_fused_add_rms(
        x, residual, weight, group_name, 1.0e-6
    )

    reduced_ref = x.clone()
    dist.all_reduce(reduced_ref)
    ref_x = reduced_ref.float() + residual.float()
    residual_ref = ref_x.to(dtype)
    variance = ref_x.pow(2).mean(dim=-1, keepdim=True)
    out_ref = (ref_x * torch.rsqrt(variance + 1.0e-6)).to(dtype) * weight

    torch.xpu.synchronize()
    out_max_diff = (out - out_ref).abs().max().float()
    residual_max_diff = (residual_out - residual_ref).abs().max().float()
    max_diff = torch.stack([out_max_diff, residual_max_diff]).max()
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            {
                "out_mean": float(out.float().mean().cpu()),
                "residual_mean": float(residual_out.float().mean().cpu()),
                "max_abs_diff": float(max_diff.cpu()),
            },
            flush=True,
        )
    if float(max_diff.cpu()) > 1.0e-3:
        raise AssertionError(f"max diff too high: {float(max_diff.cpu())}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
