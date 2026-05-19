import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Case:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    iters: int


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype name: {name}")


def default_cases() -> list[Case]:
    scale = float(os.environ.get("B70_MINIMAX_AR_ITERS_SCALE", "1.0") or "1.0")

    def scaled(value: int) -> int:
        return max(1, int(value * scale))

    return [
        Case("qk_decode_fp32", "fp32", (1, 2), scaled(5000)),
        Case("qk_prompt42_fp32", "fp32", (42, 2), scaled(5000)),
        Case("qk_profile512_fp32", "fp32", (512, 2), scaled(3000)),
        Case("hidden_decode_fp16", "fp16", (1, 3072), scaled(3000)),
        Case("hidden_prompt42_fp16", "fp16", (42, 3072), scaled(800)),
        Case("hidden_profile512_fp16", "fp16", (512, 3072), scaled(120)),
    ]


def modes() -> list[str]:
    raw = os.environ.get("B70_MINIMAX_AR_MODES")
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return ["inplace", "clone", "empty_copy"]


def run_allreduce(mode: str, x: torch.Tensor) -> torch.Tensor:
    if mode == "inplace":
        out = x
    elif mode == "clone":
        out = x.clone()
    elif mode == "empty_copy":
        out = torch.empty_like(x)
        out.copy_(x)
    else:
        raise ValueError(f"unknown mode: {mode}")
    dist.all_reduce(out)
    return out


def bench_case(mode: str, case: Case, local_rank: int) -> dict[str, object]:
    dtype = dtype_from_name(case.dtype_name)
    x = torch.ones(case.shape, dtype=dtype, device=f"xpu:{local_rank}")

    for _ in range(30):
        run_allreduce(mode, x)
    torch.xpu.synchronize()
    dist.barrier()

    avg_ms: list[float] = []
    repeats = int(os.environ.get("B70_MINIMAX_AR_REPEATS", "5") or "5")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(case.iters):
            run_allreduce(mode, x)
        torch.xpu.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        avg_ms.append(elapsed * 1000.0 / case.iters)

    nbytes = x.numel() * x.element_size()
    mean_ms = statistics.fmean(avg_ms)
    return {
        "mode": mode,
        "case": asdict(case),
        "numel": x.numel(),
        "bytes": nbytes,
        "repeats": len(avg_ms),
        "avg_ms_per_repeat": avg_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(avg_ms),
        "stdev_ms": statistics.stdev(avg_ms) if len(avg_ms) > 1 else 0.0,
        "payload_gbps": nbytes / (mean_ms / 1000.0) / 1.0e9,
    }


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    results = []
    for mode in modes():
        for case in default_cases():
            result = bench_case(mode, case, local_rank)
            if rank == 0:
                results.append(result)

    if rank == 0:
        payload = {
            "benchmark": "b70_xccl_minimax_allreduce_shapes",
            "world_size": world_size,
            "ccl_env": {
                key: os.environ[key]
                for key in sorted(os.environ)
                if key.startswith("CCL_")
            },
            "modes": modes(),
            "results": results,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
