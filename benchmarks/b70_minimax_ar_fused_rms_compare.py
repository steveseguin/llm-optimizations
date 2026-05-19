import os

import torch
import torch.distributed as dist

import minimax_ar_fused_rms_xpu  # noqa: F401


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    rows = int(os.environ.get("MINIMAX_AR_FUSED_RMS_ROWS", "1"))
    hidden_size = int(os.environ.get("MINIMAX_AR_FUSED_RMS_HIDDEN_SIZE", "3072"))
    dtype = torch.float16
    device = f"xpu:{local_rank}"
    base_seed = int(os.environ.get("MINIMAX_AR_FUSED_RMS_SEED", "2026051901"))

    torch.manual_seed(base_seed + rank)
    x = (
        torch.randn((rows, hidden_size), dtype=dtype, device=device) * 0.05
        + rank * 0.01
    ).contiguous()
    torch.manual_seed(base_seed + 999)
    residual = (
        torch.randn((rows, hidden_size), dtype=dtype, device=device) * 0.05
    ).contiguous()
    torch.manual_seed(base_seed + 1999)
    weight = (
        torch.randn((hidden_size,), dtype=dtype, device=device) * 0.05 + 1.0
    ).contiguous()

    # MiniMax delay-allreduce path:
    # rank 0 adds the replicated residual before the allreduce, then RMSNorm
    # runs on the reduced hidden state without a residual argument.
    standard_hidden = x.clone()
    if rank == 0:
        standard_hidden = standard_hidden + residual
    dist.all_reduce(standard_hidden)
    standard_float = standard_hidden.float()
    variance = standard_float.pow(2).mean(dim=-1, keepdim=True)
    standard_out = (
        (standard_float * torch.rsqrt(variance + 1.0e-6)).to(weight.dtype)
        * weight
    ).to(dtype)

    fused_out, fused_residual = (
        torch.ops.minimax_ar_fused_rms_xpu.ar_fused_add_rms(
            x, residual, weight, "0", 1.0e-6
        )
    )
    ar_input = x.clone()
    if rank == 0:
        ar_input = ar_input + residual
    ordered_out, ordered_residual = torch.ops.minimax_ar_fused_rms_xpu.ar_rms(
        ar_input.contiguous(), weight, "0", 1.0e-6
    )

    torch.xpu.synchronize()
    values = torch.tensor(
        [
            float((standard_out - fused_out).abs().max().float().cpu()),
            float((standard_out - fused_out).abs().mean().float().cpu()),
            float((standard_hidden - fused_residual).abs().max().float().cpu()),
            float((standard_out - ordered_out).abs().max().float().cpu()),
            float((standard_out - ordered_out).abs().mean().float().cpu()),
            float((standard_hidden - ordered_residual).abs().max().float().cpu()),
        ],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            {
                "rows": rows,
                "hidden_size": hidden_size,
                "max_out_diff": float(values[0].cpu()),
                "mean_out_diff": float(values[1].cpu()),
                "max_residual_diff": float(values[2].cpu()),
                "ordered_max_out_diff": float(values[3].cpu()),
                "ordered_mean_out_diff": float(values[4].cpu()),
                "ordered_max_residual_diff": float(values[5].cpu()),
            },
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
