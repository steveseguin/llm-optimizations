import argparse
import json
import os
import time

import torch
import torch.distributed as dist


MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000


def make_case(name: str, batch: int, rank: int, world_size: int, device: str):
    token_base = rank * 100000
    tokens = torch.arange(batch, dtype=torch.int64, device=device) + token_base
    if name == "rank_wins":
        values = torch.full((batch,), float(rank - 2), dtype=torch.float32, device=device)
    elif name == "mixed_sign":
        values = torch.linspace(-4.0, 4.0, batch, dtype=torch.float32, device=device)
        values = values + (rank * 0.125)
    elif name == "negative_only":
        values = torch.full((batch,), -10.0 + rank * 0.25, dtype=torch.float32, device=device)
    elif name == "token_tie":
        values = torch.ones((batch,), dtype=torch.float32, device=device)
        tokens = torch.arange(batch, dtype=torch.int64, device=device) + (world_size - rank) * 100
    elif name == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(1234 + rank)
        values = torch.randn((batch,), generator=gen, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown case: {name}")
    return values, tokens


def reference_all_gather(values: torch.Tensor, tokens: torch.Tensor, world_size: int):
    pair = torch.stack([values.float(), tokens.float()], dim=-1)
    gathered = torch.empty(
        (world_size,) + tuple(pair.shape), dtype=pair.dtype, device=pair.device
    )
    dist.all_gather_into_tensor(gathered, pair.contiguous())
    gathered = gathered.movedim(0, 1).contiguous()
    # Match torch.argmax behavior for equal values: first rank wins. The model
    # path uses lower token-id tie preference only in the packed experiment.
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=max_rank_idx).squeeze(-1).to(torch.int64)


def packed_allreduce(values: torch.Tensor, tokens: torch.Tensor):
    value_bits = values.float().view(torch.int32).to(torch.int64)
    value_bits = value_bits & MASK32
    ordered_bits = torch.where(
        (value_bits & SIGN32) != 0,
        (~value_bits) & MASK32,
        value_bits ^ SIGN32,
    )
    signed_value_key = ordered_bits - SIGN32
    tie_key = (MASK32 - tokens.to(torch.int64)) & MASK32
    packed = (signed_value_key << 32) | tie_key
    dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return (MASK32 - (packed & MASK32)).to(torch.int64), packed


def run_case(name: str, batch: int, rank: int, world_size: int, device: str, iters: int):
    values, tokens = make_case(name, batch, rank, world_size, device)
    ref = reference_all_gather(values, tokens, world_size)
    cand, packed = packed_allreduce(values, tokens)
    torch.xpu.synchronize()
    dist.barrier()

    ok = bool(torch.equal(ref, cand))
    mismatch_count = int((ref != cand).sum().item())

    start = time.perf_counter()
    for _ in range(iters):
        packed_allreduce(values, tokens)
    torch.xpu.synchronize()
    dist.barrier()
    packed_ms = (time.perf_counter() - start) * 1000.0 / iters

    start = time.perf_counter()
    for _ in range(iters):
        reference_all_gather(values, tokens, world_size)
    torch.xpu.synchronize()
    dist.barrier()
    gather_ms = (time.perf_counter() - start) * 1000.0 / iters

    return {
        "case": name,
        "ok": ok,
        "mismatch_count": mismatch_count,
        "packed_ms": packed_ms,
        "reference_all_gather_ms": gather_ms,
        "sample": {
            "rank": rank,
            "values": values[: min(8, batch)].detach().cpu().tolist(),
            "tokens": tokens[: min(8, batch)].detach().cpu().tolist(),
            "ref": ref[: min(8, batch)].detach().cpu().tolist(),
            "candidate": cand[: min(8, batch)].detach().cpu().tolist(),
            "packed": packed[: min(8, batch)].detach().cpu().tolist(),
        },
    }


def run_reduceop_smokes(rank: int, world_size: int, device: str):
    smokes = []
    for dtype in (torch.int32, torch.int64, torch.float32):
        x_max = torch.full((4,), rank + 1, dtype=dtype, device=device)
        dist.all_reduce(x_max, op=dist.ReduceOp.MAX)
        torch.xpu.synchronize()
        dist.barrier()

        x_sum = torch.full((4,), rank + 1, dtype=dtype, device=device)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        torch.xpu.synchronize()
        dist.barrier()

        max_values = x_max.detach().cpu().tolist()
        sum_values = x_sum.detach().cpu().tolist()
        smokes.append(
            {
                "dtype": str(dtype).replace("torch.", ""),
                "max_values": max_values,
                "sum_values": sum_values,
                "max_expected": world_size,
                "sum_expected": world_size * (world_size + 1) // 2,
                "max_ok": all(value == world_size for value in max_values),
                "sum_ok": all(
                    value == world_size * (world_size + 1) // 2
                    for value in sum_values
                ),
            }
        )
    return smokes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--cases",
        default="rank_wins,mixed_sign,negative_only,token_tie,random",
        help="Comma-separated probe cases.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")
    device = f"xpu:{local_rank}"

    local_smokes = run_reduceop_smokes(rank, world_size, device)
    gathered_smokes = [None] * world_size
    dist.all_gather_object(gathered_smokes, local_smokes)

    results = []
    for name in [part.strip() for part in args.cases.split(",") if part.strip()]:
        result = run_case(name, args.batch, rank, world_size, device, args.iters)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, result)
        if rank == 0:
            results.append(
                {
                    "case": name,
                    "all_ranks_ok": all(item["ok"] for item in gathered),
                    "mismatch_counts": [
                        item["mismatch_count"] for item in gathered
                    ],
                    "packed_ms_rank0": gathered[0]["packed_ms"],
                    "reference_all_gather_ms_rank0": gathered[0][
                        "reference_all_gather_ms"
                    ],
                    "rank_samples": [item["sample"] for item in gathered],
                }
            )

    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "batch": args.batch,
                    "iters": args.iters,
                    "reduceop_smokes": gathered_smokes,
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
