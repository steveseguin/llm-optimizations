#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import sys
from pathlib import Path


MODULES = {
    "logits_processor": "vllm.model_executor.layers.logits_processor",
    "gpu_model_runner": "vllm.v1.worker.gpu_model_runner",
    "minimax_m2": "vllm.model_executor.models.minimax_m2",
    "moe_wna16": "vllm.model_executor.layers.quantization.moe_wna16",
    "xpu_communicator": "vllm.distributed.device_communicators.xpu_communicator",
}


def split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def module_info(module_name: str) -> dict:
    import_stdout = io.StringIO()
    import_stderr = io.StringIO()
    with contextlib.redirect_stdout(import_stdout), contextlib.redirect_stderr(
        import_stderr
    ):
        module = importlib.import_module(module_name)
    path = Path(inspect.getfile(module)).resolve()
    text = path.read_text(errors="replace")
    return {
        "module": module_name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
        "import_stdout": import_stdout.getvalue(),
        "import_stderr": import_stderr.getvalue(),
        "markers": {
            "local_argmax_decode": "VLLM_XPU_LOCAL_ARGMAX_DECODE" in text,
            "local_argmax_gather_broadcast": (
                "VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST" in text
            ),
            "local_argmax_direct_gather": (
                "VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER" in text
            ),
            "local_argmax_pair_all_gather_label": (
                "logits.local_argmax_pair_all_gather" in text
            ),
            "minimax_logits_moe": "MINIMAX_LOGITS" in text,
            "qk_norm_restore_weight": "QK_NORM_RESTORE_WEIGHT" in text,
        },
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Write JSON diagnostics here")
    args = parser.parse_args()

    required_markers = split_env_list(os.environ.get("VLLM_RUNTIME_REQUIRE_MARKERS"))
    forbidden_markers = split_env_list(os.environ.get("VLLM_RUNTIME_FORBID_MARKERS"))
    required_any_markers = split_env_list(
        os.environ.get("VLLM_RUNTIME_REQUIRE_ANY_MARKERS")
    )
    expected_logits_path = os.environ.get("VLLM_RUNTIME_EXPECT_LOGITS_PROCESSOR")

    modules: dict[str, dict] = {}
    missing_modules: dict[str, str] = {}
    module_texts: dict[str, str] = {}
    for label, module_name in MODULES.items():
        try:
            info = module_info(module_name)
        except Exception as exc:
            missing_modules[label] = f"{type(exc).__name__}: {exc}"
            continue
        text = info.pop("text")
        module_texts[label] = text
        info["required_marker_hits"] = {
            marker: marker in text for marker in required_markers
        }
        info["forbidden_marker_hits"] = {
            marker: marker in text for marker in forbidden_markers
        }
        modules[label] = info

    errors: list[str] = []
    logits = modules.get("logits_processor")
    if logits is None:
        errors.append("logits_processor module could not be imported")
    else:
        for marker, hit in logits["required_marker_hits"].items():
            if not hit:
                errors.append(f"required marker missing from logits_processor: {marker}")
        for marker, hit in logits["forbidden_marker_hits"].items():
            if hit:
                errors.append(f"forbidden marker present in logits_processor: {marker}")
        if expected_logits_path:
            actual = str(Path(logits["path"]).resolve())
            expected = str(Path(expected_logits_path).resolve())
            if actual != expected:
                errors.append(
                    "logits_processor import path mismatch: "
                    f"actual={actual} expected={expected}"
                )
    required_any_hits = {
        marker: [
            label
            for label, text in module_texts.items()
            if marker in text
        ]
        for marker in required_any_markers
    }
    for marker, labels in required_any_hits.items():
        if not labels:
            errors.append(f"required marker missing from all inspected modules: {marker}")

    result = {
        "python": sys.executable,
        "cwd": os.getcwd(),
        "vllm_runtime_require_markers": required_markers,
        "vllm_runtime_require_any_markers": required_any_markers,
        "vllm_runtime_require_any_marker_hits": required_any_hits,
        "vllm_runtime_forbid_markers": forbidden_markers,
        "vllm_runtime_expect_logits_processor": expected_logits_path,
        "modules": {
            key: {k: v for k, v in value.items() if k != "text"}
            for key, value in modules.items()
        },
        "missing_modules": missing_modules,
        "ok": not errors,
        "errors": errors,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
