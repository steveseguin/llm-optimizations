#!/usr/bin/env python3
"""Sweep MiniMax 4096-context graph quality across prompt lengths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


BASE_CONTEXT = (
    "PCIe tensor parallel inference often spends time moving partial "
    "activations and reductions between GPUs. Software can reduce overhead "
    "with better scheduling, graph capture, fused collectives, and stable "
    "batching. "
)
FINAL_QUESTION = (
    "Final question: In eight concise numbered points, explain why four PCIe "
    "GPUs can become communication-bound during local LLM decoding and name "
    "software mitigations that preserve model quality."
)


def prompt_len(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False).input_ids)


def make_prompt_for_target(tokenizer, target_tokens: int) -> tuple[str, int, int]:
    prefix = "Use this context as background, then answer the final question. Context: "
    lo = 0
    hi = 1
    while True:
        prompt = prefix + (BASE_CONTEXT * hi) + " " + FINAL_QUESTION
        if prompt_len(tokenizer, prompt) >= target_tokens:
            break
        if hi > 4096:
            raise RuntimeError(f"could not reach target prompt length {target_tokens}")
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        prompt = prefix + (BASE_CONTEXT * mid) + " " + FINAL_QUESTION
        if prompt_len(tokenizer, prompt) >= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    repeats = lo
    prompt = prefix + (BASE_CONTEXT * repeats) + " " + FINAL_QUESTION
    return prompt, prompt_len(tokenizer, prompt), repeats


def run_target(
    args: argparse.Namespace,
    prompt_path: Path,
    out_json: Path,
    log_path: Path,
) -> tuple[int, float]:
    env = os.environ.copy()
    env.update(
        {
            "USE_LLM_SCALER_MOE": "1",
            "XPU_GRAPH": "1",
            "VLLM_XPU_ENABLE_XPU_GRAPH": "1",
            "VLLM_XPU_FORCE_GRAPH_WITH_COMM": "1",
            "VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE": "1",
            "VLLM_XPU_USE_LLM_SCALER_MOE": "1",
            "VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE": "1",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:0,1,2,3",
            "ZE_AFFINITY_MASK": "0,1,2,3",
            "CCL_ATL_TRANSPORT": "ofi",
            "CCL_TOPO_P2P_ACCESS": "1",
            "HF_HOME": "/mnt/fast-ai/llm-cache/hf",
            "TRANSFORMERS_CACHE": "/mnt/fast-ai/llm-cache/hf/transformers",
            "PYTHONPATH": (
                "/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python:"
                + env.get("PYTHONPATH", "")
            ),
            "LD_LIBRARY_PATH": (
                "/home/steve/.venvs/vllm-xpu/lib:"
                "/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/torch/lib:"
                + env.get("LD_LIBRARY_PATH", "")
            ),
        }
    )
    cmd = [
        "timeout",
        "--foreground",
        "--signal=TERM",
        "--kill-after=30s",
        args.timeout,
        sys.executable,
        "/home/steve/llm-optimizations-publish/scripts/run-vllm-minimax-quality-check.py",
        "--mode",
        "graph",
        "--model",
        args.model,
        "--out",
        str(out_json),
        "--max-tokens",
        str(args.max_tokens),
        "--runs",
        "1",
        "--prompt-file",
        str(prompt_path),
        "--tensor-parallel-size",
        "4",
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        "1",
        "--block-size",
        "256",
        "--compilation-mode",
        "none",
        "--cudagraph-mode",
        "full_decode_only",
        "--cudagraph-num-warmups",
        "0",
        "--attention-backend",
        "TRITON_ATTN",
        "--allow-nondeterministic-output",
        "--min-distinct-generated-tokens",
        "2",
        "--min-printable-nonspace-chars",
        "1",
        "--max-control-nonspace-chars",
        "0",
        "--max-nul-token-count",
        "0",
    ]
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd="/home/steve",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode, time.monotonic() - start


def summarize_result(out_json: Path, log_path: Path, returncode: int) -> dict:
    summary = {"returncode": returncode, "json": str(out_json), "log": str(log_path)}
    if out_json.exists() and out_json.stat().st_size > 0:
        try:
            data = json.loads(out_json.read_text())
            quality = data.get("quality_checks", {})
            summary.update(
                {
                    "passed": data.get("passed"),
                    "failure_reasons": data.get("failure_reasons", []),
                    "generated_tokens": quality.get("n_tokens"),
                    "distinct_generated_tokens": quality.get(
                        "distinct_generated_token_count"
                    ),
                    "nul_token_count": quality.get("nul_token_count"),
                    "control_nonspace_text_chars": quality.get(
                        "control_nonspace_text_chars"
                    ),
                    "elapsed_s": data.get("elapsed_s"),
                }
            )
        except json.JSONDecodeError as exc:
            summary.update({"passed": False, "failure_reasons": [str(exc)]})
    else:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        reasons = []
        if "RPC call to sample_tokens timed out" in text:
            reasons.append("sample_tokens timeout")
        if "No available shared memory broadcast block found" in text:
            reasons.append("shared-memory broadcast wait")
        if returncode == 124:
            reasons.append("process timeout")
        summary.update({"passed": False, "failure_reasons": reasons or ["no json"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround",
    )
    parser.add_argument(
        "--outdir",
        default="/home/steve/bench-results/minimax-m2.7-quality-gated/ctx4096-prompt-sweep",
    )
    parser.add_argument("--targets", default="128,256,512,768,1024,1280,1400")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--timeout", default="12m")
    parser.add_argument("--stop-after-fail", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    prompt_dir = outdir / "prompts"
    outdir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps({"event": "load_tokenizer", "model": args.model}),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    targets = [int(x) for x in args.targets.replace(" ", "").split(",") if x]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary_path = outdir / f"summary-{run_id}.json"
    results = []

    for target in targets:
        prompt, actual_tokens, repeats = make_prompt_for_target(tokenizer, target)
        stem = f"ctx4096-mbt{args.max_num_batched_tokens}-target{target}-actual{actual_tokens}-{run_id}"
        prompt_path = prompt_dir / f"{stem}.txt"
        out_json = outdir / f"{stem}.json"
        log_path = outdir / f"{stem}.log"
        prompt_path.write_text(prompt + "\n")
        print(
            json.dumps(
                {
                    "event": "start",
                    "target_tokens": target,
                    "actual_prompt_tokens": actual_tokens,
                    "repeats": repeats,
                    "prompt": str(prompt_path),
                    "json": str(out_json),
                    "log": str(log_path),
                }
            ),
            flush=True,
        )
        returncode, wall_s = run_target(args, prompt_path, out_json, log_path)
        result = summarize_result(out_json, log_path, returncode)
        result.update(
            {
                "target_tokens": target,
                "actual_prompt_tokens": actual_tokens,
                "repeats": repeats,
                "prompt": str(prompt_path),
                "wall_s": wall_s,
            }
        )
        results.append(result)
        summary = {
            "run_id": run_id,
            "model": args.model,
            "max_model_len": 4096,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_tokens": args.max_tokens,
            "targets": targets,
            "results": results,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"event": "result", **result}), flush=True)
        if args.stop_after_fail and not result.get("passed"):
            break

    print(json.dumps({"summary": str(summary_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
