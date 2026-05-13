#!/usr/bin/env python3
"""Run deterministic MiniMax M2.7 generation smoke checks.

This is intentionally a smoke check, not a full evaluation suite.  It verifies
that a candidate runtime is deterministic for fixed greedy prompts and records
token hashes that can be compared across graph/backend changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        "--prompt",
        action="append",
        default=None,
        help="Prompt to run. May be repeated. Defaults to a small fixed suite.",
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
        "--expected-token-sha256",
        default=None,
        help="Fail if the combined token hash differs from this value.",
    )
    return parser.parse_args()


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
        "use_inductor_graph_partition": True,
        "compile_sizes": [1],
    }
    if args.mode == "graph":
        compilation_config["cudagraph_mode"] = "PIECEWISE"

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
        disable_custom_all_reduce=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=args.enable_prefix_caching,
        compilation_config=compilation_config,
    )
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=0,
        stop_token_ids=[200020],
    )
    started = time.perf_counter()
    prompts = args.prompt or DEFAULT_PROMPTS
    run_records = []
    if args.raw_prompt:
        rendered_prompt = None
        for run_idx in range(args.runs):
            outputs = llm.generate(prompts, params)
            run_records.append(
                {
                    "run": run_idx,
                    "prompts": [
                        {
                            "prompt_index": i,
                            "n_tokens": len(output.outputs[0].token_ids),
                            "token_ids": list(output.outputs[0].token_ids),
                            "token_sha256": hashlib.sha256(
                                ",".join(map(str, output.outputs[0].token_ids)).encode()
                            ).hexdigest(),
                            "text": output.outputs[0].text,
                            "text_sha256": hashlib.sha256(
                                output.outputs[0].text.encode()
                            ).hexdigest(),
                        }
                        for i, output in enumerate(outputs)
                    ],
                }
            )
    else:
        template_path = Path(args.chat_template or Path(args.model) / "chat_template.jinja")
        rendered_prompt = template_path.read_text()
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        for run_idx in range(args.runs):
            outputs = llm.chat(
                conversations,
                params,
                chat_template=rendered_prompt,
            )
            run_records.append(
                {
                    "run": run_idx,
                    "prompts": [
                        {
                            "prompt_index": i,
                            "n_tokens": len(output.outputs[0].token_ids),
                            "token_ids": list(output.outputs[0].token_ids),
                            "token_sha256": hashlib.sha256(
                                ",".join(map(str, output.outputs[0].token_ids)).encode()
                            ).hexdigest(),
                            "text": output.outputs[0].text,
                            "text_sha256": hashlib.sha256(
                                output.outputs[0].text.encode()
                            ).hexdigest(),
                        }
                        for i, output in enumerate(outputs)
                    ],
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
    record = {
        "mode": args.mode,
        "elapsed_s": elapsed,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "n_prompts": len(prompts),
        "combined_token_sha256": combined_token_hash,
        "combined_text_sha256": combined_text_hash,
        "expected_token_sha256": args.expected_token_sha256,
        "expected_token_sha256_match": expected_match,
        "deterministic_across_runs": deterministic,
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
            "enable_prefix_caching": args.enable_prefix_caching,
            "attention_delay_allreduce": args.attention_delay_allreduce,
            "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
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
                "expected_token_sha256_match": record[
                    "expected_token_sha256_match"
                ],
            },
            indent=2,
        )
    )
    if not deterministic:
        raise SystemExit("quality smoke failed: nondeterministic token hashes")
    if expected_match is False:
        raise SystemExit("quality smoke failed: combined token hash mismatch")


if __name__ == "__main__":
    main()
