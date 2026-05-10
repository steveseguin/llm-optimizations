import json
import os
import time

import torch
import torch.distributed as dist


def parse_csv_ints(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def dtype_from_env() -> torch.dtype:
    raw = os.environ.get("B70_MINIMAX_BOUNDARY_DTYPE", "float16").lower()
    if raw in {"float16", "fp16", "half"}:
        return torch.float16
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported B70_MINIMAX_BOUNDARY_DTYPE={raw!r}")


def iters_for_tokens(tokens: int) -> int:
    override = os.environ.get("B70_MINIMAX_BOUNDARY_ITERS")
    if override:
        return int(override)
    if tokens <= 8:
        return 1000
    if tokens <= 128:
        return 300
    return 100


def sync_barrier() -> None:
    torch.xpu.synchronize()
    dist.barrier()


def run_timed(fn, iters: int) -> float:
    for _ in range(10):
        fn()
    sync_barrier()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_barrier()
    return (time.perf_counter() - start) / iters


def main() -> None:
    import minimax_qk_rms_xpu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    dtype = dtype_from_env()
    tokens_list = parse_csv_ints("B70_MINIMAX_BOUNDARY_TOKENS", "1,72,512")
    modes = parse_csv(
        "B70_MINIMAX_BOUNDARY_MODES",
        "tiny_allreduce,qk_var,qk_var_allreduce,qk_full_helper,torch_qk_reference,hidden_allreduce",
    )
    q_total = int(os.environ.get("B70_MINIMAX_Q_TOTAL", "6144"))
    kv_total = int(os.environ.get("B70_MINIMAX_KV_TOTAL", "1024"))
    hidden_total = int(os.environ.get("B70_MINIMAX_HIDDEN_TOTAL", "3072"))
    q_size = int(os.environ.get("B70_MINIMAX_Q_SIZE", str(q_total // world_size)))
    kv_size = int(os.environ.get("B70_MINIMAX_KV_SIZE", str(kv_total // world_size)))
    hidden_size = int(
        os.environ.get("B70_MINIMAX_HIDDEN_SIZE", str(hidden_total))
    )
    eps = float(os.environ.get("B70_MINIMAX_RMS_EPS", "1e-6"))

    if rank == 0:
        ccl_env = {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith("CCL_")
        }
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "dtype": str(dtype).replace("torch.", ""),
                    "q_size": q_size,
                    "kv_size": kv_size,
                    "hidden_size": hidden_size,
                    "modes": modes,
                    "ccl_env": ccl_env,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    device = torch.device(f"xpu:{local_rank}")
    for tokens in tokens_list:
        qkv = torch.randn(
            tokens, q_size + 2 * kv_size, dtype=dtype, device=device
        ).contiguous()
        q_weight = torch.randn(q_size, dtype=dtype, device=device).contiguous()
        k_weight = torch.randn(kv_size, dtype=dtype, device=device).contiguous()
        qk_var = torch.empty(tokens, 2, dtype=torch.float32, device=device)
        tiny = torch.zeros(tokens, 2, dtype=torch.float32, device=device)
        hidden = torch.zeros(tokens, hidden_size, dtype=dtype, device=device)
        q_out = torch.empty(tokens, q_size, dtype=dtype, device=device)
        k_out = torch.empty(tokens, kv_size, dtype=dtype, device=device)

        q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
        iters = iters_for_tokens(tokens)

        def tiny_allreduce() -> None:
            dist.all_reduce(tiny)

        def qk_var_only() -> None:
            torch.ops.minimax_qk_rms_xpu.var(qkv, qk_var, q_size, kv_size)

        def qk_var_allreduce() -> None:
            torch.ops.minimax_qk_rms_xpu.var(qkv, qk_var, q_size, kv_size)
            dist.all_reduce(qk_var)
            qk_var.div_(world_size)

        def qk_full_helper() -> None:
            torch.ops.minimax_qk_rms_xpu.var(qkv, qk_var, q_size, kv_size)
            dist.all_reduce(qk_var)
            qk_var.div_(world_size)
            torch.ops.minimax_qk_rms_xpu.apply(
                qkv,
                qk_var,
                q_weight,
                k_weight,
                q_out,
                k_out,
                q_size,
                kv_size,
                eps,
            )

        def torch_qk_reference() -> None:
            q_var = q.float().pow(2).mean(dim=-1, keepdim=True)
            k_var = k.float().pow(2).mean(dim=-1, keepdim=True)
            qk = torch.cat([q_var, k_var], dim=-1)
            dist.all_reduce(qk)
            q_var, k_var = (qk / world_size).chunk(2, dim=-1)
            torch.empty_like(q).copy_(
                (q.float() * torch.rsqrt(q_var + eps) * q_weight).to(dtype)
            )
            torch.empty_like(k).copy_(
                (k.float() * torch.rsqrt(k_var + eps) * k_weight).to(dtype)
            )

        def hidden_allreduce() -> None:
            dist.all_reduce(hidden)

        fns = {
            "tiny_allreduce": tiny_allreduce,
            "qk_var": qk_var_only,
            "qk_var_allreduce": qk_var_allreduce,
            "qk_full_helper": qk_full_helper,
            "torch_qk_reference": torch_qk_reference,
            "hidden_allreduce": hidden_allreduce,
        }

        for mode in modes:
            avg = run_timed(fns[mode], iters)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "tokens": tokens,
                            "iters": iters,
                            "avg_ms": avg * 1000,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
