#!/usr/bin/env python3
"""Run a small deterministic MiniMax M2.7 generation for graph checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


DEFAULT_PROMPT = (
    "You are a precise assistant. Answer the following in three short "
    "numbered points. Explain why tensor parallel inference can be "
    "communication-bound on four PCIe GPUs, and include one concrete "
    "mitigation that preserves model quality."
)


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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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
    return parser.parse_args()


def configure_env(mode: str) -> None:
    os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0,1,2,3")
    os.environ.setdefault("ZE_AFFINITY_MASK", "0,1,2,3")
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    os.environ.setdefault("CCL_TOPO_P2P_ACCESS", "1")
    os.environ.setdefault("HF_HOME", "/mnt/fast-ai/llm-cache/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{os.environ['HF_HOME']}/transformers")
    os.environ.setdefault(
        "PYTHONPATH",
        "/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python",
    )
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/home/steve/.venvs/vllm-xpu/lib:"
        "/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/torch/lib",
    )
    os.environ["VLLM_XPU_USE_LLM_SCALER_MOE"] = "1"
    if mode == "graph":
        os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"
        os.environ["VLLM_XPU_FORCE_GRAPH_WITH_COMM"] = "1"
        os.environ["VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE"] = "1"
    else:
        os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "0"
        os.environ.pop("VLLM_XPU_FORCE_GRAPH_WITH_COMM", None)
        os.environ.pop("VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE", None)


def main() -> None:
    args = parse_args()
    configure_env(args.mode)

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
        dtype="float16",
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        max_model_len=2048,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        disable_custom_all_reduce=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
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
    if args.raw_prompt:
        result = llm.generate([args.prompt], params)[0].outputs[0]
        rendered_prompt = args.prompt
    else:
        template_path = Path(args.chat_template or Path(args.model) / "chat_template.jinja")
        rendered_prompt = template_path.read_text()
        result = llm.chat(
            [[{"role": "user", "content": args.prompt}]],
            params,
            chat_template=rendered_prompt,
        )[0].outputs[0]
    elapsed = time.perf_counter() - started
    token_ids = list(result.token_ids)
    text = result.text
    record = {
        "mode": args.mode,
        "elapsed_s": elapsed,
        "max_tokens": args.max_tokens,
        "n_tokens": len(token_ids),
        "token_sha256": hashlib.sha256(
            ",".join(map(str, token_ids)).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_ids": token_ids,
        "text": text,
        "prompt": args.prompt,
        "raw_prompt": args.raw_prompt,
        "chat_template": None if args.raw_prompt else str(template_path),
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
                "n_tokens": record["n_tokens"],
                "token_sha256": record["token_sha256"],
                "text_sha256": record["text_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
