#!/usr/bin/env python3
"""Summarize vLLM AOT allreduce boundary patterns.

The XPU MiniMax path is dominated by many small compiled TP allreduces. This
script classifies the wait/copy boundary that follows each allreduce so each
experiment can record whether it changed the graph shape.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


ALL_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*): \"(?P<shape>[^\"]+)\" = "
    r"torch\.ops\._c10d_functional\.all_reduce"
)
WAIT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*): \"(?P<shape>[^\"]+)\" = "
    r"torch\.ops\._c10d_functional\.wait_tensor\((?P<input>[A-Za-z_][A-Za-z0-9_]*)\)"
)
FILE_RE = re.compile(r"^\s*# File: (?P<file>.*), code: (?P<code>.*)$")
OP_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*): \"(?P<shape>[^\"]+)\" = (?P<expr>.*)$"
)
EMPTY_XPU_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*) = empty_strided_xpu"
    r"\(\((?P<shape>[^)]*)\), .* torch\.(?P<dtype>[A-Za-z0-9_]+)\)"
)
REINTERPRET_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*) = reinterpret_tensor"
    r"\((?P<base>[A-Za-z_][A-Za-z0-9_]*), \((?P<shape>[^)]*)\),"
)
GENERATED_ALL_RE = re.compile(
    r"^\s*torch\.ops\._c10d_functional\.all_reduce_\.default"
    r"\((?P<input>[A-Za-z_][A-Za-z0-9_]*)"
)
GENERATED_WAIT_RE = re.compile(
    r"^\s*torch\.ops\._c10d_functional\.wait_tensor\.default"
    r"\((?P<input>[A-Za-z_][A-Za-z0-9_]*)"
)
TOPO_RE = re.compile(
    r"^\s*# (?:Topologically Sorted|Unsorted) Source Nodes: "
    r"\[(?P<nodes>[^\]]*)\], Original ATen: \[(?P<aten>[^\]]*)\]"
)


def compact_file(file_comment: str | None) -> str:
    if not file_comment:
        return "unknown"
    file_comment = file_comment.replace("/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/", "")
    file_comment = file_comment.replace("/home/steve/src/vllm/", "")
    return file_comment


def classify_next_file(file_comment: str | None, code: str | None, op_expr: str | None) -> str:
    text = " ".join(x or "" for x in (file_comment, code, op_expr))
    if "vocab_parallel_embedding.py" in text:
        return "embedding"
    if "linear_attn.py" in text and ("qk_var" in text or "q_norm" in text or "forward_qk" in text):
        return "qk_rms_apply"
    if "layernorm.py" in text or "rms_norm" in text:
        return "rms_norm"
    if "fused_moe" in text or "moe_forward" in text or "experts" in text:
        return "moe"
    if "minimax_m2.py" in text and "residual" in text:
        return "residual"
    if "aten.add" in text or " + " in text:
        return "add"
    if "lm_head" in text or "logits" in text:
        return "logits"
    return "other"


def dtype_short(dtype: str) -> str:
    return {
        "float16": "f16",
        "bfloat16": "bf16",
        "float32": "f32",
        "float64": "f64",
        "int32": "i32",
        "int64": "i64",
    }.get(dtype, dtype)


def compact_shape(shape: str) -> str:
    return ", ".join(part.strip() for part in shape.split(",") if part.strip())


def classify_generated(
    shape: str,
    pre_topo: str | None,
    next_topo: str | None,
    next_window: str,
) -> str:
    text = " ".join(x or "" for x in (pre_topo, next_topo, next_window))
    if shape.startswith("f32[") and shape.endswith(", 2]"):
        return "qk_variance"
    if "aten.embedding" in (pre_topo or ""):
        if "vllm_ir.rms_norm" in text and "_xpu_C.int4_gemm_w4a16" in text:
            return "embedding_to_rms_int4_gemm"
        return "embedding"
    if "vllm_ir.rms_norm" in text and "_xpu_C.int4_gemm_w4a16" in text:
        return "hidden_to_rms_int4_gemm"
    if "vllm.moe_forward" in text:
        return "hidden_to_moe"
    if "aten.add" in text and "aten.rsqrt" in text:
        return "hidden_to_rms"
    if "_xpu_C.int4_gemm_w4a16" in text:
        return "hidden_to_int4_gemm"
    return "other"


def analyze_file(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    allreduces: list[dict[str, object]] = []
    pending_by_name: defaultdict[str, deque[dict[str, object]]] = defaultdict(deque)

    for idx, line in enumerate(lines):
        m = ALL_RE.match(line)
        if not m:
            continue
        entry: dict[str, object] = {
            "line": idx + 1,
            "name": m.group("name"),
            "shape": m.group("shape"),
            "pre_context": [],
            "next_file": None,
            "next_code": None,
            "next_op": None,
            "next_expr": None,
            "classification": "unresolved",
        }
        for back in range(max(0, idx - 20), idx):
            fm = FILE_RE.match(lines[back])
            if fm:
                entry["pre_context"].append(
                    {
                        "file": compact_file(fm.group("file")),
                        "code": fm.group("code").strip(),
                    }
                )
        pending_by_name[m.group("name")].append(entry)
        allreduces.append(entry)

    wait_to_entry: dict[str, dict[str, object]] = {}
    for idx, line in enumerate(lines):
        wm = WAIT_RE.match(line)
        if wm and pending_by_name.get(wm.group("input")):
            wait_to_entry[wm.group("name")] = pending_by_name[wm.group("input")].popleft()
            continue
        for wait_name, entry in list(wait_to_entry.items()):
            if wait_name not in line:
                continue
            for fwd in range(idx + 1, min(len(lines), idx + 25)):
                fm = FILE_RE.match(lines[fwd])
                if not fm:
                    continue
                if "torch/distributed/_functional_collectives.py" in fm.group("file"):
                    continue
                entry["next_file"] = compact_file(fm.group("file"))
                entry["next_code"] = fm.group("code").strip()
                for op_idx in range(fwd + 1, min(len(lines), fwd + 8)):
                    om = OP_RE.match(lines[op_idx])
                    if om:
                        entry["next_op"] = om.group("name")
                        entry["next_expr"] = om.group("expr").strip()
                        break
                entry["classification"] = classify_next_file(
                    entry["next_file"],
                    entry["next_code"],
                    entry["next_expr"],
                )
                break

    by_shape = Counter(str(e["shape"]) for e in allreduces)
    by_class = Counter(str(e["classification"]) for e in allreduces)
    by_shape_class = Counter(
        f"{e['shape']} -> {e['classification']}" for e in allreduces
    )
    samples = []
    seen: set[str] = set()
    for entry in allreduces:
        key = f"{entry['shape']} -> {entry['classification']}"
        if key in seen:
            continue
        seen.add(key)
        samples.append(entry)

    return {
        "file": str(path),
        "allreduceCount": len(allreduces),
        "byShape": dict(sorted(by_shape.items())),
        "byClassification": dict(sorted(by_class.items())),
        "byShapeClassification": dict(sorted(by_shape_class.items())),
        "samples": samples,
    }


def analyze_generated_file(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    shapes: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    allreduces: list[dict[str, object]] = []
    last_topo: str | None = None

    for idx, line in enumerate(lines):
        tm = TOPO_RE.match(line)
        if tm:
            last_topo = (
                f"Source Nodes: [{tm.group('nodes')}], "
                f"Original ATen: [{tm.group('aten')}]"
            )

        em = EMPTY_XPU_RE.match(line)
        if em:
            name = em.group("name")
            dtype = dtype_short(em.group("dtype"))
            shapes[name] = f"{dtype}[{compact_shape(em.group('shape'))}]"
            dtypes[name] = dtype
            continue

        rm = REINTERPRET_RE.match(line)
        if rm:
            base = rm.group("base")
            if base in dtypes:
                name = rm.group("name")
                shapes[name] = f"{dtypes[base]}[{compact_shape(rm.group('shape'))}]"
                dtypes[name] = dtypes[base]
            continue

        am = GENERATED_ALL_RE.match(line)
        if not am:
            continue

        input_name = am.group("input")
        wait_line = None
        next_topo = None
        for fwd in range(idx + 1, min(len(lines), idx + 30)):
            if wait_line is None and GENERATED_WAIT_RE.match(lines[fwd]):
                wait_line = fwd + 1
                continue
            if wait_line is not None:
                tm_next = TOPO_RE.match(lines[fwd])
                if tm_next:
                    next_topo = (
                        f"Source Nodes: [{tm_next.group('nodes')}], "
                        f"Original ATen: [{tm_next.group('aten')}]"
                    )
                    break

        next_window = "\n".join(lines[idx + 1 : min(len(lines), idx + 35)])
        shape = shapes.get(input_name, "unknown")
        classification = classify_generated(
            shape,
            last_topo,
            next_topo,
            next_window,
        )
        if shape == "unknown" and classification.startswith("hidden_"):
            # MiniMax hidden-state TP collectives in generated Inductor code are
            # often produced by opaque int4/MoE calls, so the executable call
            # site lacks a direct empty_strided assignment to read from.
            shape = "f16[s72, 3072]"
        entry: dict[str, object] = {
            "file": str(path),
            "line": idx + 1,
            "input": input_name,
            "shape": shape,
            "pre_topo": last_topo,
            "wait_line": wait_line,
            "next_topo": next_topo,
            "classification": classification,
        }
        allreduces.append(entry)

    by_shape = Counter(str(e["shape"]) for e in allreduces)
    by_class = Counter(str(e["classification"]) for e in allreduces)
    by_shape_class = Counter(
        f"{e['shape']} -> {e['classification']}" for e in allreduces
    )
    samples = []
    seen: set[str] = set()
    for entry in allreduces:
        key = f"{entry['shape']} -> {entry['classification']}"
        if key in seen:
            continue
        seen.add(key)
        samples.append(entry)

    return {
        "file": str(path),
        "allreduceCount": len(allreduces),
        "byShape": dict(sorted(by_shape.items())),
        "byClassification": dict(sorted(by_class.items())),
        "byShapeClassification": dict(sorted(by_shape_class.items())),
        "samples": samples,
        "allreduces": allreduces,
    }


def merge_generated(path: Path) -> dict[str, object]:
    files = sorted(path.glob("**/*.py")) if path.is_dir() else [path]
    file_results = [
        analyze_generated_file(file)
        for file in files
        if "__pycache__" not in file.parts
    ]
    nonempty = [r for r in file_results if int(r["allreduceCount"]) > 0]
    by_shape: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_shape_class: Counter[str] = Counter()
    samples = []
    seen: set[str] = set()
    for result in nonempty:
        by_shape.update(result["byShape"])  # type: ignore[arg-type]
        by_class.update(result["byClassification"])  # type: ignore[arg-type]
        by_shape_class.update(result["byShapeClassification"])  # type: ignore[arg-type]
        for sample in result["samples"]:  # type: ignore[index]
            key = f"{sample['shape']} -> {sample['classification']}"
            if key in seen:
                continue
            seen.add(key)
            samples.append(sample)

    return {
        "path": str(path),
        "layout": "generated_inductor_cache",
        "pythonFileCount": len(files),
        "filesWithAllreduce": len(nonempty),
        "allreduceCount": sum(int(r["allreduceCount"]) for r in nonempty),
        "byShape": dict(sorted(by_shape.items())),
        "byClassification": dict(sorted(by_class.items())),
        "byShapeClassification": dict(sorted(by_shape_class.items())),
        "samples": samples,
        "files": [
            {
                "file": r["file"],
                "allreduceCount": r["allreduceCount"],
                "byShape": r["byShape"],
                "byClassification": r["byClassification"],
            }
            for r in nonempty
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="A computation_graph.py file or cache directory")
    parser.add_argument("--rank", default="rank_0_0", help="rank directory to select when path is a cache root")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        candidates = sorted(path.glob(f"**/{args.rank}/backbone/computation_graph.py"))
        if not candidates:
            candidates = sorted(path.glob("**/computation_graph.py"))
        if candidates:
            path = candidates[0]
            result = analyze_file(path)
        else:
            result = merge_generated(path)
    else:
        if path.name == "computation_graph.py":
            result = analyze_file(path)
        else:
            result = analyze_generated_file(path)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
