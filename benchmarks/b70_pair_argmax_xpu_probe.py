#!/usr/bin/env python3
"""Probe the MiniMax XPU pair-argmax helper on B70/XCCL."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

import minimax_pair_argmax_xpu


def make_case(name: str, batch: int, rank: int, world_size: int, device: str):
    tokens = torch.arange(batch, dtype=torch.int64, device=device) + rank * 100000
    if name == "rank_wins":
        values = torch.full((batch,), rank - 2.0, dtype=torch.float32, device=device)
    elif name == "mixed_sign":
        values = torch.linspace(-4.0, 4.0, batch, dtype=torch.float32, device=device)
        values = values + rank * 0.125
    elif name == "negative_only":
        values = torch.full(
            (batch,), -10.0 + rank * 0.25, dtype=torch.float32, device=device
        )
    elif name == "token_tie":
        values = torch.ones((batch,), dtype=torch.float32, device=device)
        tokens = torch.arange(batch, dtype=torch.int64, device=device) + rank * 100
    elif name == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(20260517 + rank)
        values = torch.randn((batch,), generator=gen, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown case: {name}")
    return values, tokens


def reference(values: torch.Tensor, tokens: torch.Tensor, world_size: int):
    gathered = gather_flat_pairs(values, tokens, world_size).view(
        values.shape[0], world_size, 2
    )
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(dim=-1, index=max_rank_idx).squeeze(-1).to(torch.int64)


def gather_flat_pairs(values: torch.Tensor, tokens: torch.Tensor, world_size: int):
    pair = torch.stack([values.float(), tokens.float()], dim=-1)
    gathered = torch.empty(
        (values.shape[0] * world_size, 2), dtype=pair.dtype, device=pair.device
    )
    dist.all_gather_into_tensor(gathered, pair.contiguous())
    return gathered.reshape(world_size, values.shape[0], 2).movedim(0, 1).reshape(
        values.shape[0], world_size * 2
    )


def run_timed(fn, iters: int):
    fn()
    torch.xpu.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    dist.barrier()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--cases",
        default="rank_wins,mixed_sign,negative_only,token_tie,random",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")
    # c10d functional collectives address groups by registered string name.
    from torch._C._distributed_c10d import _register_process_group

    dist.group.WORLD._set_group_name("tp")
    _register_process_group("tp", dist.group.WORLD)
    device = f"xpu:{local_rank}"

    local_rows = []
    for case in [part.strip() for part in args.cases.split(",") if part.strip()]:
        values, tokens = make_case(case, args.batch, rank, world_size, device)
        ref = reference(values, tokens, world_size)
        flat_pairs = gather_flat_pairs(values, tokens, world_size)
        reduce_only = minimax_pair_argmax_xpu.reduce_flat_pairs(
            flat_pairs.contiguous(), world_size
        )
        cand = minimax_pair_argmax_xpu.pair_argmax(
            values.contiguous(), tokens.contiguous(), "tp", world_size
        )
        torch.xpu.synchronize()
        dist.barrier()
        ok = bool(torch.equal(ref, cand))
        reduce_only_ok = bool(torch.equal(ref, reduce_only))
        mismatch_count = int((ref != cand).sum().item())
        reduce_only_mismatch_count = int((ref != reduce_only).sum().item())
        helper_ms = run_timed(
            lambda: minimax_pair_argmax_xpu.pair_argmax(
                values.contiguous(), tokens.contiguous(), "tp", world_size
            ),
            args.iters,
        )
        reduce_only_ms = run_timed(
            lambda: minimax_pair_argmax_xpu.reduce_flat_pairs(
                gather_flat_pairs(values, tokens, world_size).contiguous(),
                world_size,
            ),
            args.iters,
        )
        ref_ms = run_timed(lambda: reference(values, tokens, world_size), args.iters)
        local_rows.append(
            {
                "case": case,
                "rank": rank,
                "ok": ok,
                "mismatch_count": mismatch_count,
                "helper_ms": helper_ms,
                "reduce_only_ok": reduce_only_ok,
                "reduce_only_mismatch_count": reduce_only_mismatch_count,
                "reduce_only_ms": reduce_only_ms,
                "reference_ms": ref_ms,
                "ref_tokens": ref[: min(8, ref.numel())].detach().cpu().tolist(),
                "candidate_tokens": cand[: min(8, cand.numel())].detach().cpu().tolist(),
                "reduce_only_tokens": reduce_only[
                    : min(8, reduce_only.numel())
                ].detach().cpu().tolist(),
            }
        )

    gathered_rows = [None] * world_size
    dist.all_gather_object(gathered_rows, local_rows)
    if rank == 0:
        cases = []
        case_names = [part.strip() for part in args.cases.split(",") if part.strip()]
        for case_idx, case in enumerate(case_names):
            rows = [rank_rows[case_idx] for rank_rows in gathered_rows]
            cases.append(
                {
                    "case": case,
                    "all_ranks_ok": all(row["ok"] for row in rows),
                    "mismatch_counts": [row["mismatch_count"] for row in rows],
                    "helper_ms_by_rank": [row["helper_ms"] for row in rows],
                    "reduce_only_all_ranks_ok": all(
                        row["reduce_only_ok"] for row in rows
                    ),
                    "reduce_only_mismatch_counts": [
                        row["reduce_only_mismatch_count"] for row in rows
                    ],
                    "reduce_only_ms_by_rank": [
                        row["reduce_only_ms"] for row in rows
                    ],
                    "reference_ms_by_rank": [row["reference_ms"] for row in rows],
                    "rank0_ref_tokens": rows[0]["ref_tokens"],
                    "rank0_candidate_tokens": rows[0]["candidate_tokens"],
                    "rank0_reduce_only_tokens": rows[0]["reduce_only_tokens"],
                }
            )
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "batch": args.batch,
                    "iters": args.iters,
                    "backend": "xccl",
                    "device": "xpu",
                    "cases": cases,
                },
                indent=2,
                sort_keys=True,
            )
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
