# MiniMax Q/K RMS XPU Helper Patch Snapshot

Date: 2026-05-14

Purpose: reproduce the failed compiled-path repair branch for MiniMax M2.7 AutoRound W4A16 on 4x Intel Arc Pro B70.

Contents:

- `minimax-qk-rms-xpu-alloc-helper-20260514.patch.gz.b64`: patch against this publish repo for the XPU Q/K RMS helper extension and quality-check harness flags.
- `vllm-minimax-compiled-helper-and-trace-20260514.patch.gz.b64`: patch against `/home/steve/src/vllm` containing the MiniMax model integration, finite tracing hooks, and linear-attention tracing changes used by the diagnostics.

Reapply example:

```bash
base64 -d patches/minimax-qk-rms-xpu-alloc-helper-20260514.patch.gz.b64 | gunzip | git apply
base64 -d patches/vllm-minimax-compiled-helper-and-trace-20260514.patch.gz.b64 | gunzip | git -C /path/to/vllm apply
```

Outcome:

The helper extension builds and direct XPU numerical checks pass. It does not repair the vLLM compiled MiniMax corruption: one-token no-cudagraph probes still emit token-id `0` / NUL text. Do not use this branch as a performance result.
