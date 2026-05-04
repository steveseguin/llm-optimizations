# Qwen3.6 27B B70 Follow-Up: VDR, Root-Copy Allreduce, FP8

Date: 2026-05-04
Host: Ubuntu 24.04.4 LTS, Linux 6.17.0-22-generic
GPU: Intel Arc Pro B70 32GB, 4 cards available

## Summary

This follow-up tested three paths after the first Q4_0 tensor-parallel gains:

- single-card Q4_0 kernel constants and runtime flags;
- an env-gated SYCL Meta allreduce variant, `GGML_SYCL_COMM_ROOT_COPY=1`;
- official `Qwen/Qwen3.6-27B-FP8` Safetensors through patched vLLM/XPU.

None replaced the current best Q4_0 path. Current best quality-preserving Q4_0 GGUF result remains 3x B70 tensor parallel with single-kernel allreduce at `41.737 tok/s` decode.

## Q4_0 Single-Card Checks

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
Build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`

| Test | Result | Artifact | Decision |
| --- | ---: | --- | --- |
| `VDR_Q4_0_Q8_1_MMVQ=4` | `24.431 tok/s` | `sycl-single-vdr4-selector2-fa0-ub128-n256-20260504T140109Z.jsonl` | Neutral, not a win |
| Best `-fa`/`-ub` sweep point | `24.406 tok/s` | `sycl-single-fa-ub-sweep-selector2-20260504T140245Z.tsv` | Decode unchanged |
| `-fa 1` sweep | `24.225-24.273 tok/s` | same TSV | Slightly slower for decode |

Interpretation: graph capture, flash attention, ubatch, reorder enablement, `MMV_Y`, subgroup count, and VDR4 do not explain the Linux single-card gap versus the Windows `>27 tok/s` Q4_0 result. The next work needs to inspect the reordered Q4_0 MMVQ kernel and activation quantization/dataflow directly.

## Root-Copy Allreduce Variant

New env flag: `GGML_SYCL_COMM_ROOT_COPY=1`

Behavior: reduce partial F32 vectors on backend 0 using remote reads, then broadcast the 20 KiB reduced vector to peer devices with peer copies. This avoids scalar remote writes in the current single-kernel allreduce, but still pays one reduce kernel and peer copies per allreduce.

Important CLI detail: multi-device `-dev` must use slash-separated devices, for example `SYCL0/SYCL1/SYCL2`, not comma-separated names. Commas caused Level Zero OOM in `MUL_MAT` before allreduce evaluation.

| Mode | Selector | Result | Artifact | Decision |
| --- | --- | ---: | --- | --- |
| 2x B70 | `level_zero:0,3` | `38.259 tok/s` | `sycl-root-copy-scaling-n128-20260504T142851Z.tsv` | Slower than single-kernel allreduce |
| 3x B70 | `level_zero:2,1,3` | `39.817 tok/s` | same TSV | Slower than single-kernel allreduce |
| 4x B70 | `level_zero:3,0,1,2` | `30.371 tok/s` | same TSV | Stable but still not useful |

Interpretation: root-copy is useful as a diagnostic branch but is not a speed path. The 4-GPU bottleneck is still the 128 small 20 KiB allreduces per token and their synchronization cost.

## Official FP8 vLLM/XPU Retest

Model: `/home/steve/models/qwen3.6-27b-fp8-hf`
Engine: vLLM `0.20.1` with local XPU block-FP8 patches
Kernel path: `XPURequantFp8BlockScaledMMLinearKernel`
Env: `VLLM_XPU_BLOCK_FP8_REQUANT=1`

Current XPU kernels do not provide a native 128x128 block-FP8 W8A8 GEMM path for this checkpoint, so the local path dequantizes block-FP8 weights after load and requantizes to per-channel FP8 W8A16.

| TP | Input | Output | Latency | Output-token upper bound | Artifact |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 512 | 128 | `6.664 s` | `19.2 tok/s` | `vllm-qwen36-fp8-tp2-in512-out128-bs1-20260504T143518Z.json` |
| 2 | 512 | 512 | `25.464 s` | `20.1 tok/s` | `vllm-qwen36-fp8-tp2-in512-out512-bs1-20260504T143657Z.json` |
| 4 | 512 | 128 | `6.980 s` | `18.3 tok/s` | `vllm-qwen36-fp8-tp4-in512-out128-bs1-20260504T143317Z.json` |
| 4 | 512 | 512 | `26.606 s` | `19.2 tok/s` | `vllm-qwen36-fp8-tp4-in512-out512-bs1-20260504T143844Z.json` |

Command shape for the submitted TP2 512-output run:

```bash
VLLM_XPU_BLOCK_FP8_REQUANT=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,3 \
TP=2 INPUT_LEN=512 OUTPUT_LEN=512 BATCH_SIZE=1 MAX_MODEL_LEN=1024 GPU_MEM_UTIL=0.95 NUM_ITERS=1 WARMUP_ITERS=0 \
EXTRA_ARGS='--kv-cache-memory-bytes 1073741824 --max-num-batched-tokens 1024 --max-num-seqs 1 --enforce-eager --disable-log-stats' \
/home/steve/bench-vllm-qwen36-fp8.sh
```

LocalMaxxing submission: `cmorb75xb001ckz0489eqc9se`.

Interpretation: official FP8 is downloaded and runnable, but current vLLM/XPU FP8 is slower than Q4_0 GGUF for steady single-session decode. TP4 does not improve over TP2. FP8 should stay as a backend R&D track until native XPU block-FP8 kernels exist.
