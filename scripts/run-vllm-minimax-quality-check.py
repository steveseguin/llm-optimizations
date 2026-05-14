#!/usr/bin/env python3
"""Run MiniMax M2.7 generation smoke checks.

This is intentionally a smoke check, not a full evaluation suite. It records
token hashes for repeatability diagnostics while separating those diagnostics
from hard corruption and semantic-canary checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


DEFAULT_PROMPTS = [
    (
        "You are a precise assistant. Answer the following in three short "
        "numbered points. Explain why tensor parallel inference can be "
        "communication-bound on four PCIe GPUs, and include one concrete "
        "mitigation that preserves model quality."
    ),
    (
        "A user asks whether speculative decoding can change answer quality. "
        "Give a concise, technically accurate answer and mention one validation "
        "step before publishing a benchmark."
    ),
    (
        "Write a short Python function named median_latency_ms that accepts a "
        "list of floating point seconds and returns the median in milliseconds. "
        "Include only the function."
    ),
]


def prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name, "")
    parts = [part for part in current.split(":") if part]
    if value not in parts:
        os.environ[name] = ":".join([value, *parts])
    if name == "PYTHONPATH" and value not in sys.path:
        sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("eager", "graph"),
        required=True,
        help="Runtime path to validate.",
    )
    parser.add_argument(
        "--model",
        default="/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround",
    )
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Record generated-token top logprobs for diagnostics.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to run. May be repeated. Defaults to a small fixed suite.",
    )
    parser.add_argument(
        "--prompt-file",
        action="append",
        default=None,
        help="UTF-8 prompt file to run. May be repeated.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Use LLM.generate on the raw prompt instead of the chat template.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Chat template path. Defaults to <model>/chat_template.jinja.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM torch.compile/cudagraph execution for correctness isolation.",
    )
    parser.add_argument(
        "--disable-inductor-graph-partition",
        action="store_true",
        help="Keep torch.compile enabled but disable vLLM's inductor graph partition option.",
    )
    parser.add_argument(
        "--cudagraph-num-warmups",
        type=int,
        default=None,
        help="Override vLLM compilation_config.cudagraph_num_of_warmups.",
    )
    parser.add_argument(
        "--compile-ranges-endpoints",
        default=None,
        help=(
            "Comma-separated compilation_config.compile_ranges_endpoints. "
            "Use an empty string to force an empty list."
        ),
    )
    parser.add_argument(
        "--rms-norm-priority",
        default=None,
        help=(
            "Comma-separated vLLM IR RMSNorm provider priority, for example "
            "'xpu_kernels,native'."
        ),
    )
    parser.add_argument(
        "--compilation-mode",
        choices=("default", "none", "stock", "dynamo_once", "vllm"),
        default="default",
        help="Override vLLM compilation_config.mode for compiler-path isolation.",
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("none", "piecewise", "full", "full_decode_only", "full_and_piecewise"),
        default=None,
        help=(
            "Override vLLM compilation_config.cudagraph_mode. By default, "
            "--mode graph uses PIECEWISE."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="Override vLLM attention backend, for example TRITON_ATTN.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce. Default mirrors benchmark scripts.",
    )
    parser.add_argument(
        "--disable-llm-scaler-moe",
        action="store_true",
        help="Disable the llm-scaler MiniMax INT4 MoE path for control probes.",
    )
    parser.add_argument(
        "--allow-degenerate-output",
        action="store_true",
        help="Do not fail on all-identical tokens or control-character output.",
    )
    parser.add_argument(
        "--allow-nondeterministic-output",
        action="store_true",
        help="Record but do not fail if repeated greedy runs produce different token hashes.",
    )
    parser.add_argument(
        "--min-distinct-generated-tokens",
        type=int,
        default=2,
        help="Fail if generated token diversity is below this threshold.",
    )
    parser.add_argument(
        "--min-printable-nonspace-chars",
        type=int,
        default=1,
        help="Fail if generated printable non-space text chars are below this threshold.",
    )
    parser.add_argument(
        "--max-control-nonspace-chars",
        type=int,
        default=0,
        help="Fail if generated non-space control chars exceed this threshold.",
    )
    parser.add_argument(
        "--max-nul-token-count",
        type=int,
        default=0,
        help="Fail if generated token id 0 count exceeds this threshold.",
    )
    parser.add_argument(
        "--require-substring",
        action="append",
        default=None,
        help=(
            "Substring that must appear in the combined generated text. May be "
            "repeated. Use with canary prompts."
        ),
    )
    parser.add_argument(
        "--require-regex",
        action="append",
        default=None,
        help=(
            "Python regex that must match the combined generated text. May be "
            "repeated. Use with canary prompts."
        ),
    )
    parser.add_argument(
        "--async-scheduling",
        choices=("default", "on", "off"),
        default="default",
        help="Override vLLM async scheduling for determinism/isolation tests.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="Disable prefix caching.",
    )
    parser.set_defaults(enable_prefix_caching=False)
    parser.add_argument(
        "--attention-delay-allreduce",
        action="store_true",
        default=True,
        help="Enable the MiniMax attention delayed-allreduce patch.",
    )
    parser.add_argument(
        "--no-attention-delay-allreduce",
        dest="attention_delay_allreduce",
        action="store_false",
    )
    parser.add_argument("--vllm-cache-root", default=None)
    parser.add_argument(
        "--finite-trace-file",
        default=None,
        help=(
            "Enable local XPU finite tensor tracing and write JSONL records to "
            "this path. Intended for tiny correctness probes only."
        ),
    )
    parser.add_argument(
        "--finite-trace-limit",
        type=int,
        default=8,
        help="Maximum records per finite-trace tag when --finite-trace-file is set.",
    )
    parser.add_argument(
        "--preserve-cudagraph-warmups",
        action="store_true",
        help=(
            "Set local XPU patch env so vLLM does not rewrite an explicit "
            "cudagraph_num_of_warmups value to 1."
        ),
    )
    parser.add_argument(
        "--disable-auto-compile-ranges",
        action="store_true",
        help=(
            "Set local XPU patch env to skip auto-adding max_num_batched_tokens "
            "to compilation_config.compile_ranges_endpoints."
        ),
    )
    parser.add_argument(
        "--eager-for-uncovered-compile-ranges",
        action="store_true",
        help=(
            "Set local XPU patch env so piecewise compilation falls back to eager "
            "for runtime shapes outside compile_sizes/compile_ranges."
        ),
    )
    parser.add_argument(
        "--skip-compiled-prefill",
        action="store_true",
        help=(
            "Set local XPU patch env so vLLM bypasses the compiled model wrapper "
            "for prefill/mixed steps with more than one token."
        ),
    )
    parser.add_argument(
        "--split-qk-var-allreduce",
        action="store_true",
        help=(
            "Use separate Q and K variance all-reduces in MiniMax Q/K RMSNorm "
            "instead of the default concatenated qk_var collective."
        ),
    )
    parser.add_argument(
        "--eager-qk-norm",
        action="store_true",
        help=(
            "Force MiniMax Q/K RMSNorm through a torch.compiler.disable helper "
            "for compiled-path correctness isolation."
        ),
    )
    parser.add_argument(
        "--decomposed-qk-norm",
        action="store_true",
        help=(
            "Use a decomposed MiniMax Q/K RMSNorm expression with explicit "
            "contiguous boundaries to avoid unsafe compile fusion."
        ),
    )
    parser.add_argument(
        "--expected-token-sha256",
        default=None,
        help="Fail if the combined token hash differs from this value.",
    )
    return parser.parse_args()


def serialize_logprobs(logprobs):
    if logprobs is None:
        return None
    serialized = []
    for step in logprobs:
        if step is None:
            serialized.append(None)
            continue
        entries = []
        for token_id, item in step.items():
            record = {"token_id": int(token_id)}
            for attr in ("logprob", "rank", "decoded_token"):
                if hasattr(item, attr):
                    value = getattr(item, attr)
                    if attr == "logprob" and value is not None:
                        value = float(value)
                        if not math.isfinite(value):
                            record["logprob_nonfinite"] = str(value)
                            value = None
                    record[attr] = value
            entries.append(record)
        entries.sort(key=lambda entry: entry.get("rank") or 10**9)
        serialized.append(entries)
    return serialized


def text_quality_stats(token_ids: list[int], text: str) -> dict:
    distinct_tokens = sorted(set(token_ids))
    printable_chars = sum(
        1 for char in text if char.isprintable() and not char.isspace()
    )
    control_nonspace_chars = sum(
        1 for char in text if not char.isprintable() and not char.isspace()
    )
    nul_token_count = sum(1 for token in token_ids if token == 0)
    return {
        "n_tokens": len(token_ids),
        "distinct_generated_token_count": len(distinct_tokens),
        "first_distinct_generated_tokens": distinct_tokens[:16],
        "printable_nonspace_text_chars": printable_chars,
        "control_nonspace_text_chars": control_nonspace_chars,
        "nul_token_count": nul_token_count,
        "nontrivial_tokens": len(distinct_tokens) > 1,
        "nontrivial_text": printable_chars > 0,
        "control_char_output": control_nonspace_chars > 0 or nul_token_count > 0,
    }


def semantic_requirement_results(
    generated_text: str,
    required_substrings: list[str] | None,
    required_regexes: list[str] | None,
) -> dict:
    substring_results = [
        {"substring": substring, "matched": substring in generated_text}
        for substring in (required_substrings or [])
    ]
    regex_results = []
    for pattern in required_regexes or []:
        try:
            matched = re.search(pattern, generated_text, flags=re.MULTILINE) is not None
            regex_results.append({"regex": pattern, "matched": matched})
        except re.error as exc:
            regex_results.append(
                {"regex": pattern, "matched": False, "error": str(exc)}
            )
    return {
        "required_substrings": substring_results,
        "required_regexes": regex_results,
        "all_required_substrings_matched": all(
            item["matched"] for item in substring_results
        ),
        "all_required_regexes_matched": all(item["matched"] for item in regex_results),
    }


def prompt_output_record(prompt_index: int, output) -> dict:
    token_ids = list(output.outputs[0].token_ids)
    text = output.outputs[0].text
    return {
        "prompt_index": prompt_index,
        "n_tokens": len(token_ids),
        "token_ids": token_ids,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest(),
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "logprobs": serialize_logprobs(output.outputs[0].logprobs),
        "quality": text_quality_stats(token_ids, text),
    }


def configure_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1,2,3")
    os.environ.setdefault("ZE_AFFINITY_MASK", "0,1,2,3")
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    os.environ.setdefault("CCL_TOPO_P2P_ACCESS", "1")
    os.environ.setdefault("HF_HOME", "/mnt/fast-ai/llm-cache/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{os.environ['HF_HOME']}/transformers")
    prepend_env_path(
        "PYTHONPATH",
        "/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python",
    )
    prepend_env_path("LD_LIBRARY_PATH", "/home/steve/.venvs/vllm-xpu/lib")
    prepend_env_path(
        "LD_LIBRARY_PATH",
        "/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/torch/lib",
    )
    if args.vllm_cache_root:
        os.environ["VLLM_CACHE_ROOT"] = args.vllm_cache_root
    if args.finite_trace_file:
        os.environ["VLLM_XPU_FINITE_TRACE"] = "1"
        os.environ["VLLM_XPU_FINITE_TRACE_FILE"] = args.finite_trace_file
        os.environ["VLLM_XPU_FINITE_TRACE_LIMIT"] = str(args.finite_trace_limit)
    else:
        os.environ.pop("VLLM_XPU_FINITE_TRACE", None)
        os.environ.pop("VLLM_XPU_FINITE_TRACE_FILE", None)
        os.environ.pop("VLLM_XPU_FINITE_TRACE_LIMIT", None)
    if args.preserve_cudagraph_warmups:
        os.environ["VLLM_XPU_PRESERVE_CUDAGRAPH_WARMUPS"] = "1"
    else:
        os.environ.pop("VLLM_XPU_PRESERVE_CUDAGRAPH_WARMUPS", None)
    if args.disable_auto_compile_ranges:
        os.environ["VLLM_XPU_DISABLE_AUTO_COMPILE_RANGES"] = "1"
    else:
        os.environ.pop("VLLM_XPU_DISABLE_AUTO_COMPILE_RANGES", None)
    if args.eager_for_uncovered_compile_ranges:
        os.environ["VLLM_XPU_EAGER_FOR_UNCOVERED_COMPILE_RANGES"] = "1"
    else:
        os.environ.pop("VLLM_XPU_EAGER_FOR_UNCOVERED_COMPILE_RANGES", None)
    if args.skip_compiled_prefill:
        os.environ["VLLM_XPU_SKIP_COMPILED_PREFILL"] = "1"
    else:
        os.environ.pop("VLLM_XPU_SKIP_COMPILED_PREFILL", None)
    if args.split_qk_var_allreduce:
        os.environ["VLLM_MINIMAX_QK_VAR_SPLIT_ALLREDUCE"] = "1"
    else:
        os.environ.pop("VLLM_MINIMAX_QK_VAR_SPLIT_ALLREDUCE", None)
    if args.eager_qk_norm:
        os.environ["VLLM_MINIMAX_QK_NORM_EAGER"] = "1"
    else:
        os.environ.pop("VLLM_MINIMAX_QK_NORM_EAGER", None)
    if args.decomposed_qk_norm:
        os.environ["VLLM_MINIMAX_QK_NORM_DECOMPOSED"] = "1"
    else:
        os.environ.pop("VLLM_MINIMAX_QK_NORM_DECOMPOSED", None)
    if args.disable_llm_scaler_moe:
        os.environ["VLLM_XPU_USE_LLM_SCALER_MOE"] = "0"
    else:
        os.environ["VLLM_XPU_USE_LLM_SCALER_MOE"] = "1"
    if args.attention_delay_allreduce:
        os.environ["VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE"] = "1"
    else:
        os.environ.pop("VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE", None)
    if args.mode == "graph":
        os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"
        os.environ["VLLM_XPU_FORCE_GRAPH_WITH_COMM"] = "1"
        os.environ["VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE"] = "1"
    else:
        os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "0"
        os.environ.pop("VLLM_XPU_FORCE_GRAPH_WITH_COMM", None)
        os.environ.pop("VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE", None)


def main() -> None:
    args = parse_args()
    configure_env(args)

    from vllm import LLM, SamplingParams

    compilation_config = {
        "compile_sizes": [1],
    }
    if args.compilation_mode != "default":
        mode_map = {
            "none": 0,
            "stock": 1,
            "dynamo_once": 2,
            "vllm": 3,
        }
        compilation_config["mode"] = mode_map[args.compilation_mode]
    if not args.disable_inductor_graph_partition:
        compilation_config["use_inductor_graph_partition"] = True
    if args.cudagraph_mode is not None:
        compilation_config["cudagraph_mode"] = args.cudagraph_mode.upper()
    elif args.mode == "graph":
        compilation_config["cudagraph_mode"] = "PIECEWISE"
    if args.cudagraph_num_warmups is not None:
        compilation_config["cudagraph_num_of_warmups"] = args.cudagraph_num_warmups
    if args.compile_ranges_endpoints is not None:
        endpoints = [
            int(part)
            for part in args.compile_ranges_endpoints.replace(" ", "").split(",")
            if part
        ]
        compilation_config["compile_ranges_endpoints"] = endpoints

    llm_kwargs = {}
    if args.rms_norm_priority:
        llm_kwargs["ir_op_priority"] = {
            "rms_norm": [
                part
                for part in args.rms_norm_priority.replace(" ", "").split(",")
                if part
            ]
        }
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.async_scheduling != "default":
        llm_kwargs["async_scheduling"] = args.async_scheduling == "on"

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="mp",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        enable_chunked_prefill=True,
        enable_prefix_caching=args.enable_prefix_caching,
        compilation_config=compilation_config,
        attention_backend=args.attention_backend,
        **llm_kwargs,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=0,
        stop_token_ids=[200020],
        logprobs=args.logprobs,
    )
    started = time.perf_counter()
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        prompts.extend(Path(path).read_text().rstrip("\n") for path in args.prompt_file)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    run_records = []
    if args.raw_prompt:
        rendered_prompt = None
        for run_idx in range(args.runs):
            prompt_records = []
            for i, prompt in enumerate(prompts):
                outputs = llm.generate([prompt], params)
                prompt_records.append(prompt_output_record(i, outputs[0]))
            run_records.append(
                {
                    "run": run_idx,
                    "prompts": prompt_records,
                }
            )
    else:
        template_path = Path(args.chat_template or Path(args.model) / "chat_template.jinja")
        rendered_prompt = template_path.read_text()
        for run_idx in range(args.runs):
            prompt_records = []
            for i, prompt in enumerate(prompts):
                outputs = llm.chat(
                    [[{"role": "user", "content": prompt}]],
                    params,
                    chat_template=rendered_prompt,
                )
                prompt_records.append(prompt_output_record(i, outputs[0]))
            run_records.append(
                {
                    "run": run_idx,
                    "prompts": prompt_records,
                }
            )
    elapsed = time.perf_counter() - started
    combined_tokens = []
    combined_text_parts = []
    for run in run_records:
        for prompt_record in run["prompts"]:
            combined_tokens.extend(prompt_record["token_ids"])
            combined_tokens.append(-1)
            combined_text_parts.append(prompt_record["text"])
            combined_text_parts.append("\n---\n")
    combined_token_hash = hashlib.sha256(
        ",".join(map(str, combined_tokens)).encode()
    ).hexdigest()
    combined_text_hash = hashlib.sha256("".join(combined_text_parts).encode()).hexdigest()
    generated_tokens = [
        token
        for run in run_records
        for prompt_record in run["prompts"]
        for token in prompt_record["token_ids"]
    ]
    generated_text = "".join(
        prompt_record["text"]
        for run in run_records
        for prompt_record in run["prompts"]
    )
    quality_stats = text_quality_stats(generated_tokens, generated_text)
    semantic_checks = semantic_requirement_results(
        generated_text,
        args.require_substring,
        args.require_regex,
    )
    nontrivial_tokens = (
        quality_stats["distinct_generated_token_count"]
        >= args.min_distinct_generated_tokens
    )
    nontrivial_text = (
        quality_stats["printable_nonspace_text_chars"]
        >= args.min_printable_nonspace_chars
    )
    control_char_output = (
        quality_stats["control_nonspace_text_chars"]
        > args.max_control_nonspace_chars
        or quality_stats["nul_token_count"] > args.max_nul_token_count
    )
    degenerate_output = (
        not nontrivial_tokens or not nontrivial_text or control_char_output
    )
    first_run_hashes = [p["token_sha256"] for p in run_records[0]["prompts"]]
    deterministic = all(
        [p["token_sha256"] for p in run["prompts"]] == first_run_hashes
        for run in run_records[1:]
    )
    expected_match = (
        None
        if args.expected_token_sha256 is None
        else combined_token_hash == args.expected_token_sha256
    )
    failure_reasons = []
    if not deterministic and not args.allow_nondeterministic_output:
        failure_reasons.append("nondeterministic token hashes")
    if expected_match is False:
        failure_reasons.append("combined token hash mismatch")
    if degenerate_output and not args.allow_degenerate_output:
        failure_reasons.append("degenerate or corrupt generated output")
    if not semantic_checks["all_required_substrings_matched"]:
        failure_reasons.append("required substring missing")
    if not semantic_checks["all_required_regexes_matched"]:
        failure_reasons.append("required regex missing or invalid")
    record = {
        "mode": args.mode,
        "elapsed_s": elapsed,
            "max_tokens": args.max_tokens,
            "logprobs": args.logprobs,
        "runs": args.runs,
        "n_prompts": len(prompts),
        "combined_token_sha256": combined_token_hash,
        "combined_text_sha256": combined_text_hash,
        "expected_token_sha256": args.expected_token_sha256,
        "expected_token_sha256_match": expected_match,
        "deterministic_across_runs": deterministic,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "quality_checks": {
            **quality_stats,
            "nontrivial_tokens": nontrivial_tokens,
            "nontrivial_text": nontrivial_text,
            "control_char_output": control_char_output,
            "degenerate_output": degenerate_output,
            "min_distinct_generated_tokens": args.min_distinct_generated_tokens,
            "min_printable_nonspace_chars": args.min_printable_nonspace_chars,
            "max_control_nonspace_chars": args.max_control_nonspace_chars,
            "max_nul_token_count": args.max_nul_token_count,
            "allow_degenerate_output": args.allow_degenerate_output,
            "allow_nondeterministic_output": args.allow_nondeterministic_output,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "llm_scaler_moe": not args.disable_llm_scaler_moe,
            "semantic_checks": semantic_checks,
        },
        "run_records": run_records,
        "prompts": prompts,
        "raw_prompt": args.raw_prompt,
        "chat_template": None if args.raw_prompt else str(template_path),
        "runtime": {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "block_size": args.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "disable_inductor_graph_partition": args.disable_inductor_graph_partition,
            "enable_prefix_caching": args.enable_prefix_caching,
            "attention_delay_allreduce": args.attention_delay_allreduce,
            "preserve_cudagraph_warmups": args.preserve_cudagraph_warmups,
            "disable_auto_compile_ranges": args.disable_auto_compile_ranges,
            "eager_for_uncovered_compile_ranges": (
                args.eager_for_uncovered_compile_ranges
            ),
            "skip_compiled_prefill": args.skip_compiled_prefill,
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "rms_norm_priority": args.rms_norm_priority,
            "compilation_mode": args.compilation_mode,
            "cudagraph_mode": args.cudagraph_mode,
            "attention_backend": args.attention_backend,
            "async_scheduling": args.async_scheduling,
            "cudagraph_num_warmups": args.cudagraph_num_warmups,
            "compile_ranges_endpoints": args.compile_ranges_endpoints,
        },
        "compilation_config": compilation_config,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": record["mode"],
                "elapsed_s": record["elapsed_s"],
                "runs": record["runs"],
                "n_prompts": record["n_prompts"],
                "combined_token_sha256": record["combined_token_sha256"],
                "combined_text_sha256": record["combined_text_sha256"],
                "deterministic_across_runs": record[
                    "deterministic_across_runs"
                ],
                "passed": record["passed"],
                "failure_reasons": record["failure_reasons"],
                "quality_checks": record["quality_checks"],
                "expected_token_sha256_match": record[
                    "expected_token_sha256_match"
                ],
            },
            indent=2,
        )
    )
    if failure_reasons:
        raise SystemExit(
            "quality smoke failed: " + "; ".join(failure_reasons)
        )


if __name__ == "__main__":
    main()
