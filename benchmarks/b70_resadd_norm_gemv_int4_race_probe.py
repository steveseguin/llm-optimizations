#!/usr/bin/env python3
"""Probe llm-scaler fused ResAdd/RMSNorm/INT4 GEMV correctness on B70.

This is a synthetic MiniMax-shape check. It compares:

1. vLLM XPU oneDNN `int4_gemm_w4a16` fed by a separately computed RMSNorm.
2. llm-scaler `esimd_resadd_norm_gemv_int4_pert`.

The fused llm-scaler op launches one workgroup per output row but mutates
`residual` from only output-row workgroup 0. Other output-row workgroups may
still be reading `residual`, which makes this op unsafe as a drop-in MiniMax
projection fusion unless the residual/norm producer is separated or the launch
topology is redesigned.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401
from custom_esimd_kernels_vllm import esimd_resadd_norm_gemv_int4_pert


GROUP_SIZE = 128
PACK_FACTOR = 8


@dataclass
class ShapeResult:
    n: int
    k: int
    vllm_mean_abs: float
    vllm_max_abs: float
    vllm_rel: float
    vllm_cos: float
    fused_mean_abs: float
    fused_max_abs: float
    fused_rel: float
    fused_cos: float
    fused_vs_vllm_mean_abs: float
    fused_vs_vllm_max_abs: float


def quantize_int4(weight_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight_fp16.shape
    assert k % GROUP_SIZE == 0 and k % PACK_FACTOR == 0
    groups = k // GROUP_SIZE
    grouped = weight_fp16.float().numpy().reshape(n, groups, GROUP_SIZE)
    max_abs = np.abs(grouped).max(axis=2)
    scales = np.where(max_abs > 0, max_abs / 7.0, 1.0).astype(np.float16)
    quantized = (
        np.round(grouped / scales[:, :, None].astype(np.float32))
        .clip(-8, 7)
        .astype(np.int32)
        + 8
    )
    qpack = quantized.reshape(n, k).reshape(n, k // PACK_FACTOR, PACK_FACTOR)
    qpack = qpack.astype(np.uint32)
    packed = np.zeros((n, k // PACK_FACTOR), dtype=np.uint32)
    for bit in range(PACK_FACTOR):
        packed |= (qpack[:, :, bit] & 0xF) << (bit * 4)
    return torch.from_numpy(packed.view(np.int32)), torch.from_numpy(scales)


def dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    qw = qweight.numpy().view(np.uint32)
    sc = scales.numpy().astype(np.float32)
    unpacked = np.zeros((n, k), dtype=np.float32)
    for bit in range(PACK_FACTOR):
        unpacked[:, bit::PACK_FACTOR] = ((qw >> (bit * 4)) & 0xF).astype(np.float32) - 8.0
    unpacked = unpacked.reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    return torch.from_numpy((unpacked * sc[:, :, None]).reshape(n, k)).half()


def compare_tensor(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float, float]:
    diff = (out.float() - ref.float()).abs()
    rel = diff.mean().item() / (ref.abs().mean().item() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=-1).item()
    return diff.mean().item(), diff.max().item(), rel, cos


def run_shape(n: int, k: int, device: str, seed: int) -> ShapeResult:
    torch.manual_seed(seed + n + k)
    eps = 1e-6

    hidden = torch.randn(1, k, dtype=torch.float16, device=device)
    residual = torch.randn(1, k, dtype=torch.float16, device=device)
    norm_weight = torch.randn(k, dtype=torch.float16, device=device) * 0.1 + 1.0
    weight_fp16 = torch.randn(n, k, dtype=torch.float16)

    qweight_cpu, scales_cpu = quantize_int4(weight_fp16)
    dequant = dequantize_int4(qweight_cpu, scales_cpu, n, k).to(device)
    qweight = qweight_cpu.to(device)
    scales = scales_cpu.to(device)

    added = hidden + residual
    normed = added * torch.rsqrt(torch.mean(added * added, dim=-1, keepdim=True) + eps) * norm_weight
    ref = (normed.float() @ dequant.float().t()).cpu()

    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    vllm_out = torch.ops._xpu_C.int4_gemm_w4a16(
        normed, qweight.t(), None, scales.t().contiguous(), qzeros, GROUP_SIZE, None
    ).cpu()

    fused_out = torch.empty(1, n, dtype=torch.float16, device=device)
    normed_out = torch.empty(1, k, dtype=torch.float16, device=device)
    residual_copy = residual.clone()
    esimd_resadd_norm_gemv_int4_pert(
        hidden, residual_copy, norm_weight, qweight, scales, fused_out, normed_out, eps
    )
    torch.xpu.synchronize()
    fused_cpu = fused_out.cpu()

    v_mean, v_max, v_rel, v_cos = compare_tensor(vllm_out, ref)
    f_mean, f_max, f_rel, f_cos = compare_tensor(fused_cpu, ref)
    vf_diff = (fused_cpu.float() - vllm_out.float()).abs()
    return ShapeResult(
        n=n,
        k=k,
        vllm_mean_abs=v_mean,
        vllm_max_abs=v_max,
        vllm_rel=v_rel,
        vllm_cos=v_cos,
        fused_mean_abs=f_mean,
        fused_max_abs=f_max,
        fused_rel=f_rel,
        fused_cos=f_cos,
        fused_vs_vllm_mean_abs=vf_diff.mean().item(),
        fused_vs_vllm_max_abs=vf_diff.max().item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    shapes = [
        (2048, 3072),  # MiniMax TP4 qkv projection
        (3072, 1536),  # MiniMax TP4 o_proj shape
        (512, 3072),
        (1536, 3072),
        (256, 3072),
    ]
    results = [run_shape(n, k, args.device, args.seed) for n, k in shapes]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return

    for result in results:
        print(
            f"N={result.n} K={result.k} "
            f"vllm_rel={result.vllm_rel:.6f} fused_rel={result.fused_rel:.6f} "
            f"fused_vs_vllm_mean={result.fused_vs_vllm_mean_abs:.5f} "
            f"fused_vs_vllm_max={result.fused_vs_vllm_max_abs:.5f}"
        )


if __name__ == "__main__":
    main()
