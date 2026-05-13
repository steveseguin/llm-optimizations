#!/usr/bin/env python3
"""Summarize repeated vLLM benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--target-output-tok-s", type=float, default=73.30631164343902)
    parser.add_argument("--min-ratio", type=float, default=0.985)
    parser.add_argument("--max-cv-pct", type=float, default=1.0)
    parser.add_argument("--label", default="minimax-m2.7-current-best")
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def main() -> None:
    args = parse_args()
    runs = []
    for path in args.json_files:
        data = json.loads(path.read_text())
        elapsed = float(data["elapsed_time"])
        output_tokens = args.output_len * args.num_prompts
        prompt_tokens = args.input_len * args.num_prompts
        total_tokens = prompt_tokens + output_tokens
        runs.append(
            {
                "json": str(path),
                "elapsed_s": elapsed,
                "tok_s_out": output_tokens / elapsed,
                "tok_s_total_from_lengths": total_tokens / elapsed,
                "tok_s_total_reported": float(data["tokens_per_second"]),
                "num_requests": data.get("num_requests"),
                "total_num_tokens_reported": data.get("total_num_tokens"),
            }
        )

    outs = [run["tok_s_out"] for run in runs]
    totals = [run["tok_s_total_reported"] for run in runs]
    out_mean = statistics.mean(outs)
    out_std = sample_std(outs)
    out_cv_pct = 0.0 if math.isclose(out_mean, 0.0) else out_std / out_mean * 100.0
    min_required = args.target_output_tok_s * args.min_ratio
    failures = []
    if min(outs) < min_required:
        failures.append(
            f"min output tok/s {min(outs):.6f} below required {min_required:.6f}"
        )
    if out_cv_pct > args.max_cv_pct:
        failures.append(
            f"output tok/s CV {out_cv_pct:.3f}% above allowed {args.max_cv_pct:.3f}%"
        )

    record = {
        "label": args.label,
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "target_output_tok_s": args.target_output_tok_s,
            "min_ratio": args.min_ratio,
            "min_required_output_tok_s": min_required,
            "max_cv_pct": args.max_cv_pct,
        },
        "shape": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
        },
        "summary": {
            "runs": len(runs),
            "tok_s_out_mean": out_mean,
            "tok_s_out_median": statistics.median(outs),
            "tok_s_out_min": min(outs),
            "tok_s_out_max": max(outs),
            "tok_s_out_stddev": out_std,
            "tok_s_out_cv_pct": out_cv_pct,
            "tok_s_total_reported_mean": statistics.mean(totals),
            "tok_s_total_reported_median": statistics.median(totals),
            "tok_s_total_reported_min": min(totals),
            "tok_s_total_reported_max": max(totals),
        },
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    if failures:
        print(json.dumps({"failures": failures}, indent=2, sort_keys=True))
        raise SystemExit("repeatability thresholds failed")


if __name__ == "__main__":
    main()
