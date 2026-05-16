#!/usr/bin/env python3
"""Census MiniMax M2.7 TP collectives in vLLM computation_graph.py files.

The MiniMax decode graph has a stable collective sequence: one embedding
allreduce, then 62 layer triplets of Q/K RMS variance, attention o_proj hidden,
and MoE hidden allreduces. This script classifies by the generated all_reduce_N
index instead of nearby source comments so negative or positive optimization
screens can compare exact collective structure.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ALLREDUCE_RE = re.compile(
    r"^\s*(?P<name>all_reduce(?:_(?P<idx>\d+))?): "
    r"\"(?P<shape>[^\"]+)\" = torch\.ops\._c10d_functional\.all_reduce"
)
WAIT_RE = re.compile(
    r"^\s*(?P<name>wait_tensor(?:_(?P<idx>\d+))?): "
    r"\"(?P<shape>[^\"]+)\" = torch\.ops\._c10d_functional\.wait_tensor"
    r"\((?P<input>all_reduce(?:_\d+)?)\)"
)
COPY_RE = re.compile(
    r"^\s*(?P<name>copy_(?:_\d+)?|copy__\d+): "
    r"\"(?P<shape>[^\"]+)\" = (?P<target>[A-Za-z_][A-Za-z0-9_]*)\.copy_"
    r"\((?P<wait>wait_tensor(?:_\d+)?)\)"
)


def category_for_index(idx: int | None) -> str:
    if idx is None:
        return "embedding_hidden"
    mod = idx % 3
    if mod == 1:
        return "qk_rms_variance"
    if mod == 2:
        return "attention_o_proj_hidden"
    return "moe_hidden"


def analyze_file(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    waits: dict[str, tuple[int, str]] = {}
    copies: dict[str, tuple[int, str, str]] = {}
    for line_no, line in enumerate(lines, start=1):
        if match := WAIT_RE.match(line):
            waits[match.group("input")] = (line_no, match.group("name"))
        if match := COPY_RE.match(line):
            copies[match.group("wait")] = (
                line_no,
                match.group("name"),
                match.group("target"),
            )

    entries: list[dict[str, object]] = []
    for line_no, line in enumerate(lines, start=1):
        match = ALLREDUCE_RE.match(line)
        if not match:
            continue
        idx_text = match.group("idx")
        idx = int(idx_text) if idx_text is not None else None
        name = match.group("name")
        wait_line, wait_name = waits.get(name, (None, None))
        copy_line, copy_name, copy_target = copies.get(
            wait_name or "", (None, None, None)
        )
        entries.append(
            {
                "name": name,
                "index": idx,
                "category": category_for_index(idx),
                "shape": match.group("shape"),
                "allreduce_line": line_no,
                "wait_name": wait_name,
                "wait_line": wait_line,
                "wait_gap_lines": (
                    wait_line - line_no if wait_line is not None else None
                ),
                "copy_name": copy_name,
                "copy_target": copy_target,
                "copy_target_is_clone": (
                    str(copy_target).startswith("clone")
                    if copy_target is not None
                    else None
                ),
                "copy_line": copy_line,
                "copy_gap_lines": (
                    copy_line - line_no if copy_line is not None else None
                ),
            }
        )

    by_category = Counter(str(e["category"]) for e in entries)
    by_shape = Counter(str(e["shape"]) for e in entries)
    by_category_shape = Counter(
        f"{e['category']} -> {e['shape']}" for e in entries
    )
    wait_gaps = Counter(
        str(e["wait_gap_lines"]) for e in entries if e["wait_gap_lines"] is not None
    )
    copy_gaps = Counter(
        str(e["copy_gap_lines"]) for e in entries if e["copy_gap_lines"] is not None
    )
    copy_target_kinds = Counter(
        "clone" if e["copy_target_is_clone"] else "original_or_temp"
        for e in entries
        if e["copy_target_is_clone"] is not None
    )

    return {
        "file": str(path),
        "allreduce_count": len(entries),
        "wait_count": sum(1 for e in entries if e["wait_line"] is not None),
        "copy_count": sum(1 for e in entries if e["copy_line"] is not None),
        "by_category": dict(sorted(by_category.items())),
        "by_shape": dict(sorted(by_shape.items())),
        "by_category_shape": dict(sorted(by_category_shape.items())),
        "wait_gap_lines": dict(sorted(wait_gaps.items())),
        "copy_gap_lines": dict(sorted(copy_gaps.items())),
        "copy_target_kinds": dict(sorted(copy_target_kinds.items())),
        "first_entries": entries[:6],
        "last_entries": entries[-6:],
    }


def find_graphs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("**/rank_*_0/backbone/computation_graph.py"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="A cache hash dir or graph file.")
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    files = find_graphs(args.path)
    results = [analyze_file(path) for path in files]
    aggregate_category: Counter[str] = Counter()
    aggregate_shape: Counter[str] = Counter()
    aggregate_category_shape: Counter[str] = Counter()
    aggregate_wait_gaps: Counter[str] = Counter()
    aggregate_copy_gaps: Counter[str] = Counter()
    aggregate_copy_target_kinds: Counter[str] = Counter()
    failures: list[str] = []

    for result in results:
        aggregate_category.update(result["by_category"])  # type: ignore[arg-type]
        aggregate_shape.update(result["by_shape"])  # type: ignore[arg-type]
        aggregate_category_shape.update(
            result["by_category_shape"]  # type: ignore[arg-type]
        )
        aggregate_wait_gaps.update(result["wait_gap_lines"])  # type: ignore[arg-type]
        aggregate_copy_gaps.update(result["copy_gap_lines"])  # type: ignore[arg-type]
        aggregate_copy_target_kinds.update(
            result["copy_target_kinds"]  # type: ignore[arg-type]
        )
        if result["allreduce_count"] != 187:
            failures.append(f"{result['file']}: allreduce_count={result['allreduce_count']}")
        if result["wait_count"] != result["allreduce_count"]:
            failures.append(f"{result['file']}: wait_count={result['wait_count']}")
        if result["copy_count"] != result["allreduce_count"]:
            failures.append(f"{result['file']}: copy_count={result['copy_count']}")
        if result["by_category"] != {
            "attention_o_proj_hidden": 62,
            "embedding_hidden": 1,
            "moe_hidden": 62,
            "qk_rms_variance": 62,
        }:
            failures.append(f"{result['file']}: category mismatch")

    if len(results) != args.expected_ranks:
        failures.append(f"rank file count={len(results)} expected={args.expected_ranks}")

    output = {
        "path": str(args.path),
        "rank_file_count": len(results),
        "passed": not failures,
        "failures": failures,
        "aggregate": {
            "allreduce_count": sum(int(r["allreduce_count"]) for r in results),
            "wait_count": sum(int(r["wait_count"]) for r in results),
            "copy_count": sum(int(r["copy_count"]) for r in results),
            "by_category": dict(sorted(aggregate_category.items())),
            "by_shape": dict(sorted(aggregate_shape.items())),
            "by_category_shape": dict(sorted(aggregate_category_shape.items())),
            "wait_gap_lines": dict(sorted(aggregate_wait_gaps.items())),
            "copy_gap_lines": dict(sorted(aggregate_copy_gaps.items())),
            "copy_target_kinds": dict(sorted(aggregate_copy_target_kinds.items())),
        },
        "ranks": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.strict and failures:
        raise SystemExit("MiniMax AOT computation graph census failed")


if __name__ == "__main__":
    main()
