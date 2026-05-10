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
        if not candidates:
            raise SystemExit(f"no computation_graph.py found under {path}")
        path = candidates[0]

    print(json.dumps(analyze_file(path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
