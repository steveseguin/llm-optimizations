#!/usr/bin/env python3
"""Add fused Qwen3.5 beta-alpha projection tensors to a GGUF file.

The input GGUF is preserved. The output GGUF contains every original tensor plus
one F32 tensor named `blk.N.ssm_ba.weight` for each layer that has both
`blk.N.ssm_beta.weight` and `blk.N.ssm_alpha.weight`.

The fused tensor layout stores beta/alpha rows interleaved inside each SSM
group. The row order is chosen so the existing qwen3next-style fused graph
emits the same flat beta and alpha vectors as Qwen3.5's separate tensors.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def import_gguf(llama_cpp: Path):
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf  # type: ignore
    from gguf.gguf_reader import GGUFReader  # type: ignore

    return gguf, GGUFReader


def field_value_and_subtype(field, gguf):
    value = field.contents()
    subtype = None
    if field.types and field.types[0] == gguf.GGUFValueType.ARRAY:
        subtype = field.types[-1]
    return value, subtype


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", type=Path, default=Path("/home/steve/src/llama.cpp-q4-b70"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}")

    gguf, GGUFReader = import_gguf(args.llama_cpp)
    reader = GGUFReader(args.input)

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise SystemExit("input GGUF has no general.architecture")
    arch = arch_field.contents()

    writer = gguf.GGUFWriter(args.output, arch)
    writer.data_alignment = reader.alignment

    n_k_heads_field = reader.get_field(f"{arch}.ssm.group_count")
    n_v_heads_field = reader.get_field(f"{arch}.ssm.time_step_rank")
    if n_k_heads_field is None or n_v_heads_field is None:
        raise SystemExit(f"input GGUF is missing {arch} SSM head metadata")
    n_k_heads = int(n_k_heads_field.contents())
    n_v_heads = int(n_v_heads_field.contents())
    if n_v_heads % n_k_heads != 0:
        raise SystemExit(f"n_v_heads must be divisible by n_k_heads, got {n_v_heads}/{n_k_heads}")
    head_ratio = n_v_heads // n_k_heads

    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue
        value, subtype = field_value_and_subtype(field, gguf)
        writer.add_key_value(key, value, field.types[0], sub_type=subtype)

    tensors = {t.name: t for t in reader.tensors}
    fused_names = set(tensors)
    alpha_re = re.compile(r"^(blk\.\d+)\.ssm_alpha\.weight$")
    fused_count = 0

    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
            tensor_endianess=reader.endianess,
        )

        m = alpha_re.match(tensor.name)
        if not m:
            continue

        prefix = m.group(1)
        beta_name = f"{prefix}.ssm_beta.weight"
        fused_name = f"{prefix}.ssm_ba.weight"
        if fused_name in fused_names:
            continue
        if beta_name not in tensors:
            raise SystemExit(f"missing beta tensor for {tensor.name}: {beta_name}")

        beta = tensors[beta_name]
        alpha = tensor
        if beta.tensor_type != gguf.GGMLQuantizationType.F32 or alpha.tensor_type != gguf.GGMLQuantizationType.F32:
            raise SystemExit(f"expected F32 alpha/beta for {prefix}, got {beta.tensor_type}/{alpha.tensor_type}")
        if beta.data.shape != alpha.data.shape:
            raise SystemExit(f"shape mismatch for {prefix}: beta {beta.data.shape}, alpha {alpha.data.shape}")

        beta_by_group = beta.data.reshape(n_k_heads, head_ratio, beta.data.shape[1])
        alpha_by_group = alpha.data.reshape(n_k_heads, head_ratio, alpha.data.shape[1])
        fused = np.empty((n_k_heads, 2 * head_ratio, beta.data.shape[1]), dtype=np.float32)
        fused[:, :head_ratio, :] = beta_by_group
        fused[:, head_ratio:, :] = alpha_by_group
        fused = np.ascontiguousarray(fused.reshape(2 * n_v_heads, beta.data.shape[1]))
        writer.add_tensor(fused_name, fused)
        fused_names.add(fused_name)
        fused_count += 1

    if fused_count == 0:
        raise SystemExit("no fused tensors were added")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    print(f"wrote {args.output}")
    print(f"added fused tensors: {fused_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
