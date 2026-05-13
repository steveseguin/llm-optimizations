#!/usr/bin/env python3
"""Classify c10d collectives in a vLLM/Inductor AOT cache.

The generated Inductor cache can contain both FX-comment call sites and actual
Python calls. This script summarizes both forms so MiniMax allreduce/fence work
can be tracked across cache hashes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


COMMENT_RE = re.compile(
    r'Tensor "([^"]+)".*= call_function'
    r'\[target=torch\.ops\._c10d_functional\.all_reduce_\.default\]'
)
CALL_RE = re.compile(
    r"torch\.ops\._c10d_functional\.all_reduce_\.default\(([^,\)]+)"
)
WAIT_RE = re.compile(
    r"torch\.ops\._c10d_functional\.wait_tensor\.default\(([^,\)]+)"
)
EMPTY_XPU_RE = re.compile(
    r"(?P<buf>[A-Za-z_][A-Za-z0-9_]*) = empty_strided_xpu"
    r"\((?P<shape>\([^\)]*\)), (?P<stride>\([^\)]*\)), torch\.(?P<dtype>\w+)\)"
)
SOURCE_RE = re.compile(r"Topologically Sorted Source Nodes: \[([^\]]+)\]")
ORIGINAL_ATEN_RE = re.compile(r"Original ATen: \[([^\]]+)\]")
ALLREDUCE_SOURCE_INDEX_RE = re.compile(r"^all_reduce(?:_(\d+))?$")


def parse_source_nodes(line: str) -> str | None:
    if match := SOURCE_RE.search(line):
        return " | ".join(x.strip() for x in match.group(1).split(","))
    return None


def parse_original_aten(line: str) -> str | None:
    if match := ORIGINAL_ATEN_RE.search(line):
        return " | ".join(x.strip() for x in match.group(1).split(","))
    return None


def normalize_shape(raw: str) -> str:
    raw = " ".join(raw.split())
    raw = raw.replace("[", "[").replace("]", "]")
    return raw


def classify_allreduce(source_context: str, shape: str) -> str:
    """Infer the model boundary for MiniMax M2.7 TP collectives.

    The decode AOT graph emits one embedding collective, then a repeating
    62-layer triplet:
      all_reduce_(3L+1): Q/K RMS variance, fp32[*, 2]
      all_reduce_(3L+2): attention o_proj hidden state, fp16[*, 3072]
      all_reduce_(3L+3): MoE hidden state, fp16[*, 3072]
    Shape is used as a guard where the source numbering is absent.
    """
    if source_context == "embedding | all_reduce" or "embedding" in source_context:
        return "embedding_hidden"
    match = ALLREDUCE_SOURCE_INDEX_RE.match(source_context)
    if match:
        if match.group(1) is None:
            return "embedding_hidden"
        idx = int(match.group(1))
        mod = idx % 3
        if mod == 1:
            return "qk_rms_variance"
        if mod == 2:
            return "attention_o_proj_hidden"
        return "moe_hidden"
    if "(1, 2)" in shape or "(s72, 2)" in shape:
        return "qk_rms_variance"
    if "(1, 3072)" in shape or "(s72, 3072)" in shape:
        return "hidden_or_embedding"
    return "unknown"


def surrounding_source(
    lines: list[str],
    start: int,
    stop: int,
    *,
    reverse: bool = False,
) -> tuple[str, str]:
    span = range(max(0, start), min(len(lines), stop))
    if reverse:
        span = reversed(list(span))
    for i in span:
        source = parse_source_nodes(lines[i])
        if source is None:
            continue
        if "all_reduce" in source or "wait_tensor" in source:
            continue
        aten = parse_original_aten(lines[i]) or "unknown"
        return source, aten
    return "unknown", "unknown"


def classify_cache(cache: Path) -> dict:
    comment_shapes: Counter[str] = Counter()
    actual_shapes: Counter[str] = Counter()
    actual_source_contexts: Counter[str] = Counter()
    actual_categories: Counter[str] = Counter()
    producer_contexts: Counter[str] = Counter()
    consumer_contexts: Counter[str] = Counter()
    source_nodes: Counter[str] = Counter()
    actual_calls = 0
    actual_waits = 0
    immediate_waits = 0
    wait_gaps: Counter[int] = Counter()
    files_with_calls: Counter[str] = Counter()
    examples: list[dict] = []

    for path in sorted(cache.rglob("*.py")):
        rel = str(path.relative_to(cache))
        lines = path.read_text(errors="replace").splitlines()
        buffer_shapes: dict[str, str] = {}

        for line in lines:
            if match := EMPTY_XPU_RE.search(line):
                buffer_shapes[match.group("buf")] = (
                    f"torch.{match.group('dtype')}{match.group('shape')}"
                    f" stride={match.group('stride')}"
                )
            if match := COMMENT_RE.search(line):
                comment_shapes[normalize_shape(match.group(1))] += 1
            if match := SOURCE_RE.search(line):
                nodes = tuple(x.strip() for x in match.group(1).split(","))
                if any("all_reduce" in node for node in nodes):
                    source_nodes[" | ".join(nodes)] += 1

        for idx, line in enumerate(lines):
            call = CALL_RE.search(line)
            if call:
                actual_calls += 1
                files_with_calls[rel] += 1
                buf = call.group(1).strip()
                actual_shapes[buffer_shapes.get(buf, "unknown")] += 1
                source_context = "unknown"
                for prev_line in reversed(lines[max(0, idx - 4) : idx]):
                    source = parse_source_nodes(prev_line)
                    if source:
                        source_context = source
                        break
                category = classify_allreduce(
                    source_context, buffer_shapes.get(buf, "unknown")
                )
                producer_source, producer_aten = surrounding_source(
                    lines, idx - 28, idx, reverse=True
                )
                actual_source_contexts[source_context] += 1
                actual_categories[category] += 1
                producer_contexts[
                    f"{category}: {producer_source} :: {producer_aten}"
                ] += 1
                for gap, next_line in enumerate(lines[idx + 1 : idx + 8], start=1):
                    wait = WAIT_RE.search(next_line)
                    if wait and wait.group(1).strip() == buf:
                        immediate_waits += 1
                        wait_gaps[gap] += 1
                        consumer_source, consumer_aten = surrounding_source(
                            lines, idx + gap + 1, idx + gap + 31
                        )
                        consumer_contexts[
                            f"{category}: {consumer_source} :: {consumer_aten}"
                        ] += 1
                        if len(examples) < 12:
                            examples.append(
                                {
                                    "file": rel,
                                    "line": idx + 1,
                                    "buffer": buf,
                                    "category": category,
                                    "shape": buffer_shapes.get(buf, "unknown"),
                                    "source_context": source_context,
                                    "producer_source": producer_source,
                                    "producer_aten": producer_aten,
                                    "consumer_source": consumer_source,
                                    "consumer_aten": consumer_aten,
                                    "wait_gap_lines": gap,
                                    "call": line.strip(),
                                    "wait": next_line.strip(),
                                }
                            )
                        break
            if WAIT_RE.search(line):
                actual_waits += 1

    return {
        "cache": str(cache),
        "fx_comment_allreduce_shapes": dict(comment_shapes.most_common()),
        "actual_allreduce_input_shapes": dict(actual_shapes.most_common()),
        "actual_allreduce_source_contexts": dict(
            actual_source_contexts.most_common()
        ),
        "actual_allreduce_categories": dict(actual_categories.most_common()),
        "producer_contexts": dict(producer_contexts.most_common()),
        "consumer_contexts": dict(consumer_contexts.most_common()),
        "source_node_comments": dict(source_nodes.most_common()),
        "actual_allreduce_call_lines": actual_calls,
        "actual_wait_tensor_call_lines": actual_waits,
        "actual_allreduce_wait_pairs_within_7_lines": immediate_waits,
        "wait_gap_lines": {str(k): v for k, v in sorted(wait_gaps.items())},
        "files_with_actual_calls": dict(files_with_calls.most_common()),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.cache.is_dir():
        raise SystemExit(f"cache directory not found: {args.cache}")

    result = classify_cache(args.cache)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"cache={result['cache']}")
    print(f"actual_allreduce_call_lines={result['actual_allreduce_call_lines']}")
    print(f"actual_wait_tensor_call_lines={result['actual_wait_tensor_call_lines']}")
    print(
        "actual_allreduce_wait_pairs_within_7_lines="
        f"{result['actual_allreduce_wait_pairs_within_7_lines']}"
    )
    print("wait_gap_lines=" + json.dumps(result["wait_gap_lines"], sort_keys=True))
    print("\nfx_comment_allreduce_shapes:")
    for shape, count in result["fx_comment_allreduce_shapes"].items():
        print(f"{count:5d}  {shape}")
    print("\nactual_allreduce_input_shapes:")
    for shape, count in result["actual_allreduce_input_shapes"].items():
        print(f"{count:5d}  {shape}")
    print("\nactual_allreduce_source_contexts:")
    for context, count in result["actual_allreduce_source_contexts"].items():
        print(f"{count:5d}  {context}")
    print("\nactual_allreduce_categories:")
    for category, count in result["actual_allreduce_categories"].items():
        print(f"{count:5d}  {category}")
    print("\nproducer_contexts:")
    for context, count in result["producer_contexts"].items():
        print(f"{count:5d}  {context}")
    print("\nconsumer_contexts:")
    for context, count in result["consumer_contexts"].items():
        print(f"{count:5d}  {context}")
    print("\nfiles_with_actual_calls:")
    for path, count in result["files_with_actual_calls"].items():
        print(f"{count:5d}  {path}")


if __name__ == "__main__":
    main()
