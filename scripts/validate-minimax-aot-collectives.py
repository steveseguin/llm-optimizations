#!/usr/bin/env python3
"""Validate MiniMax M2.7 AOT graph collectives for quality-preserving TP4."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


EXPECTED_CATEGORIES = {
    "embedding_hidden": 8,
    "qk_rms_variance": 496,
    "attention_o_proj_hidden": 496,
    "moe_hidden": 496,
}


def load_classifier(script_dir: Path):
    path = script_dir / "classify-vllm-aot-collectives.py"
    spec = importlib.util.spec_from_file_location("classify_vllm_aot_collectives", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load classifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path, help="Inductor cache directory.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--expected-allreduces", type=int, default=1496)
    parser.add_argument("--expected-waits", type=int, default=1496)
    parser.add_argument("--expected-wait-gap", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = load_classifier(Path(__file__).resolve().parent)
    result = classifier.classify_cache(args.cache)

    failures: list[str] = []
    if result["actual_allreduce_call_lines"] != args.expected_allreduces:
        failures.append(
            "actual_allreduce_call_lines="
            f"{result['actual_allreduce_call_lines']} expected={args.expected_allreduces}"
        )
    if result["actual_wait_tensor_call_lines"] != args.expected_waits:
        failures.append(
            "actual_wait_tensor_call_lines="
            f"{result['actual_wait_tensor_call_lines']} expected={args.expected_waits}"
        )
    if (
        result["actual_allreduce_wait_pairs_within_7_lines"]
        != args.expected_allreduces
    ):
        failures.append(
            "actual_allreduce_wait_pairs_within_7_lines="
            f"{result['actual_allreduce_wait_pairs_within_7_lines']} "
            f"expected={args.expected_allreduces}"
        )
    expected_wait_gaps = {str(args.expected_wait_gap): args.expected_allreduces}
    if result["wait_gap_lines"] != expected_wait_gaps:
        failures.append(
            f"wait_gap_lines={result['wait_gap_lines']} expected={expected_wait_gaps}"
        )
    if result["actual_allreduce_categories"] != EXPECTED_CATEGORIES:
        failures.append(
            "actual_allreduce_categories="
            f"{result['actual_allreduce_categories']} expected={EXPECTED_CATEGORIES}"
        )

    record = {
        "passed": not failures,
        "failures": failures,
        "expected_categories": EXPECTED_CATEGORIES,
        "classification": result,
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(
        json.dumps(
            {
                "passed": record["passed"],
                "failures": failures,
                "actual_allreduce_call_lines": result[
                    "actual_allreduce_call_lines"
                ],
                "actual_wait_tensor_call_lines": result[
                    "actual_wait_tensor_call_lines"
                ],
                "actual_allreduce_categories": result[
                    "actual_allreduce_categories"
                ],
                "wait_gap_lines": result["wait_gap_lines"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict and failures:
        raise SystemExit("MiniMax AOT collective validation failed")


if __name__ == "__main__":
    main()
