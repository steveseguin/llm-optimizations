# 2026-05-06 - FP8 TP2 two-card group check

## Context

Goal: evaluate whether Qwen3.6 27B FP8 can run as a practical two-card
tensor-parallel group on 2x Arc Pro B70, leaving the other two cards for a
second group in a 2x2 serving layout.

Models tested:

- Static compressed-tensors FP8:
  `/home/steve/models/qwen3.6-27b-fp8-vrfai`
- Dynamic FP8 HF shard set:
  `/home/steve/models/qwen3.6-27b-fp8-hf`

Important runtime rule: vLLM/Torch runs must keep
`/home/steve/.venvs/vllm-xpu-managed/lib` first in `LD_LIBRARY_PATH` and should
not be launched from a oneAPI `setvars.sh` shell.

## Findings

### Static compressed-tensors FP8, TP2/PP1

Result: does not load.

Command shape:

```bash
MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai
ONEAPI_DEVICE_SELECTOR=level_zero:0,1
TP=2 PP=1 QUANTIZATION=compressed-tensors
INPUT_LEN=512 OUTPUT_LEN=128 MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.80
/home/steve/bench-vllm-qwen36-fp8.sh
```

Even with `GPU_MEM_UTIL=0.70` and `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`,
model construction hit XPU OOM during parameter allocation.

Logs:

- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp1-in512-out128-bs1-20260506T103027Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp1-in512-out128-bs1-20260506T103146Z.log`

### Dynamic FP8, TP2/PP1, current default block-FP8 fallback

Result: does not load.

The current local XPU block-FP8 fallback dequantizes 128x128 block-FP8 weights
to BF16 in `XPUBF16Fp8BlockScaledMMLinearKernel.process_weights_after_loading`.
TP2 runs fail during the BF16 temporary/final materialization path, for example:

```text
weight_bf16 = weight.to(torch.bfloat16) * expanded_scale
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 170.00 MiB.
```

Logs:

- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out128-bs1-20260506T103227Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out512-bs1-20260506T103518Z.log`

### Dynamic FP8, TP2/PP1, opt-in XPU requant path

Result: loads, but is not a speed path.

The local `VLLM_XPU_BLOCK_FP8_REQUANT=1` path selected
`XPURequantFp8BlockScaledMMLinearKernel`, converting block-FP8 checkpoint
weights into XPU W8A16 FP8 GEMM format. This is not bit-equivalent to the
checkpoint and should be treated as a quality-risk experiment until evaluated.

No-spec smoke:

- 512 prompt / 128 output
- average latency: `14.068391455002711 s`
- output speed: `9.098410 tok/s`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out128-bs1-20260506T103808Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out128-bs1-20260506T103808Z.log`

N-gram speculative run:

- 512 prompt / 512 output
- `num_speculative_tokens=4`, lookup min/max `2/4`
- average latency: `20.898872508500062 s`
- output speed: `24.498929 tok/s`
- total speed: `48.997859 tok/s`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out512-bs1-20260506T103946Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-fp8-tp2-pp1-in512-out512-bs1-20260506T103946Z.log`

## Interpretation

TP2/PP1 is currently useful only as a capacity/debug path, not as the desired
2x2 performance layout:

- Static compressed-tensors FP8 TP2/PP1 does not fit.
- Dynamic FP8 TP2/PP1 does not fit with the exact BF16 fallback.
- Dynamic FP8 TP2/PP1 can fit with local requantization, but the best measured
  run is only `24.498929 tok/s` and changes the block-FP8 quantization semantics.

The current validated FP8 speed path remains TP4/PP1 on all four B70s with
compressed-tensors FP8 and CPU n-gram speculation at `49.581893 tok/s` for
512/512. The current validated quality-preserving GGUF path remains Q4_0 TP3 at
`46.194319 tok/s`.

## Next

- Do not submit these TP2 requant runs to LocalMaxxing as recommended results.
- Keep 2x2 serving on hold until either:
  - a native XPU block-scaled FP8 GEMM path exists, or
  - the requant path passes a quality/eval check and improves materially.
- If we keep working this track, register `VLLM_XPU_BLOCK_FP8_REQUANT` as a
  known vLLM env var and add per-layer memory tracing around block-FP8 kernel
  selection and post-load conversion.
