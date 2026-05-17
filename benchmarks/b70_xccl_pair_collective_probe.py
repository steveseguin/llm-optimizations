#!/usr/bin/env python3
"""Probe tiny pair-collective argmax strategies on B70/XCCL.

The MiniMax TP4 local-argmax path needs each rank to exchange a tiny
`(float32 max_logit, float32 global_token_id)` tensor per generated token.
This probe keeps the correctness oracle as a full all-gather reference and
times alternate collectives before any candidate is wired into vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist


PairFn = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class Case:
    name: str
    batch: int


def make_pair(case: str, batch: int, rank: int, world_size: int, device: str):
    tokens = torch.arange(batch, dtype=torch.int64, device=device) + rank * 100000
    if case == "rank_wins":
        values = torch.full((batch,), rank - 2.0, dtype=torch.float32, device=device)
    elif case == "mixed_sign":
        values = torch.linspace(-4.0, 4.0, batch, dtype=torch.float32, device=device)
        values = values + rank * 0.125
    elif case == "negative_only":
        values = torch.full((batch,), -10.0 + rank * 0.25, dtype=torch.float32, device=device)
    elif case == "token_tie":
        values = torch.ones((batch,), dtype=torch.float32, device=device)
        tokens = torch.arange(batch, dtype=torch.int64, device=device) + rank * 100
    elif case == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(20260517 + rank)
        values = torch.randn((batch,), generator=gen, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unknown case: {case}")
    return torch.stack([values, tokens.float()], dim=-1)


def reduce_pair(gathered: torch.Tensor) -> torch.Tensor:
    # gathered shape: [batch, world_size, 2]
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top_tokens.squeeze(-1).to(torch.int64)


def all_gather_into_tensor_pair(local_pair: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = torch.empty(
        (world_size,) + tuple(local_pair.shape),
        dtype=local_pair.dtype,
        device=local_pair.device,
    )
    dist.all_gather_into_tensor(gathered, local_pair.contiguous())
    return gathered.movedim(0, 1).contiguous()


def all_gather_list_pair(local_pair: torch.Tensor, world_size: int) -> torch.Tensor:
    gather_list = [torch.empty_like(local_pair) for _ in range(world_size)]
    dist.all_gather(gather_list, local_pair.contiguous())
    return torch.stack(gather_list, dim=0).movedim(0, 1).contiguous()


def gather_broadcast_pair(local_pair: torch.Tensor, world_size: int) -> torch.Tensor:
    rank = dist.get_rank()
    root = 0
    gather_list = [torch.empty_like(local_pair) for _ in range(world_size)] if rank == root else None
    dist.gather(local_pair.contiguous(), gather_list=gather_list, dst=root)
    top_tokens_i32 = torch.empty((local_pair.shape[0],), dtype=torch.int32, device=local_pair.device)
    if rank == root:
        assert gather_list is not None
        gathered = torch.stack(gather_list, dim=0).movedim(0, 1).contiguous()
        top_tokens_i32.copy_(reduce_pair(gathered).to(torch.int32))
    dist.broadcast(top_tokens_i32, src=root)
    # Return a synthetic gathered tensor so the common reducer can validate the
    # selected token without special casing the output shape.
    out = torch.empty((local_pair.shape[0], world_size, 2), dtype=local_pair.dtype, device=local_pair.device)
    out[:, :, 0] = -float("inf")
    out[:, 0, 0] = 0.0
    out[:, 0, 1] = top_tokens_i32.float()
    return out


def all_to_all_repeated_pair(local_pair: torch.Tensor, world_size: int) -> torch.Tensor:
    # Repeats the tiny local pair once per destination. This sends more bytes
    # than all_gather, but exercises a different XCCL path for a tiny payload.
    send = local_pair.unsqueeze(0).expand(world_size, *local_pair.shape).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return recv.movedim(0, 1).contiguous()


def time_fn(fn: PairFn, local_pair: torch.Tensor, world_size: int, iters: int) -> float:
    # Warm up each path once so the measured loop is not the first collective.
    fn(local_pair, world_size)
    torch.xpu.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        fn(local_pair, world_size)
    torch.xpu.synchronize()
    dist.barrier()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_candidate(
    name: str,
    fn: PairFn,
    local_pair: torch.Tensor,
    ref_tokens: torch.Tensor,
    world_size: int,
    iters: int,
):
    try:
        gathered = fn(local_pair, world_size)
        torch.xpu.synchronize()
        dist.barrier()
        tokens = reduce_pair(gathered)
        ok = bool(torch.equal(tokens, ref_tokens))
        mismatch_count = int((tokens != ref_tokens).sum().item())
        elapsed_ms = time_fn(fn, local_pair, world_size, iters)
        return {
            "name": name,
            "ok": ok,
            "mismatch_count": mismatch_count,
            "avg_ms": elapsed_ms,
            "sample_tokens": tokens[: min(8, tokens.numel())].detach().cpu().tolist(),
            "error": None,
        }
    except Exception as exc:
        try:
            dist.barrier()
        except Exception:
            pass
        return {
            "name": name,
            "ok": False,
            "mismatch_count": None,
            "avg_ms": None,
            "sample_tokens": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--cases",
        default="rank_wins,mixed_sign,negative_only,token_tie,random",
    )
    parser.add_argument(
        "--candidates",
        default="all_gather_into_tensor,all_gather_list,gather_broadcast,all_to_all_repeated",
        help="Comma-separated candidate collective names to run.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")
    device = f"xpu:{local_rank}"

    all_candidates: dict[str, PairFn] = {
        "all_gather_into_tensor": all_gather_into_tensor_pair,
        "all_gather_list": all_gather_list_pair,
        "gather_broadcast": gather_broadcast_pair,
        "all_to_all_repeated": all_to_all_repeated_pair,
    }
    candidate_names = [
        part.strip() for part in args.candidates.split(",") if part.strip()
    ]
    unknown_candidates = [name for name in candidate_names if name not in all_candidates]
    if unknown_candidates:
        raise ValueError(f"unknown candidates: {unknown_candidates}")
    candidates: list[tuple[str, PairFn]] = [
        (name, all_candidates[name]) for name in candidate_names
    ]

    local_results = []
    for case in [part.strip() for part in args.cases.split(",") if part.strip()]:
        local_pair = make_pair(case, args.batch, rank, world_size, device)
        ref_gathered = all_gather_into_tensor_pair(local_pair, world_size)
        ref_tokens = reduce_pair(ref_gathered)
        case_result = {
            "case": case,
            "rank": rank,
            "reference_tokens": ref_tokens[: min(8, ref_tokens.numel())].detach().cpu().tolist(),
            "candidates": [
                run_candidate(name, fn, local_pair, ref_tokens, world_size, args.iters)
                for name, fn in candidates
            ],
        }
        local_results.append(case_result)

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        cases = []
        for case_idx, case_name in enumerate(
            [part.strip() for part in args.cases.split(",") if part.strip()]
        ):
            rows = [rank_rows[case_idx] for rank_rows in gathered_results]
            candidate_names = [item["name"] for item in rows[0]["candidates"]]
            cases.append(
                {
                    "case": case_name,
                    "reference_rank0": rows[0]["reference_tokens"],
                    "candidates": [
                        {
                            "name": candidate_name,
                            "all_ranks_ok": all(
                                row["candidates"][cand_idx]["ok"] for row in rows
                            ),
                            "mismatch_counts": [
                                row["candidates"][cand_idx]["mismatch_count"]
                                for row in rows
                            ],
                            "avg_ms_by_rank": [
                                row["candidates"][cand_idx]["avg_ms"] for row in rows
                            ],
                            "errors": [
                                row["candidates"][cand_idx]["error"] for row in rows
                            ],
                            "rank0_tokens": rows[0]["candidates"][cand_idx][
                                "sample_tokens"
                            ],
                        }
                        for cand_idx, candidate_name in enumerate(candidate_names)
                    ],
                }
            )
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "batch": args.batch,
                    "iters": args.iters,
                    "candidate_names": candidate_names,
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
