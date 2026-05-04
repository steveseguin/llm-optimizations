# Qwen3.6 27B B70 follow-up: Q8 activation cache, FP8 artifacts, MTP blocker

Date: 2026-05-04
Host: Ubuntu 24.04.4 LTS, Intel Compute Runtime 26.14.37833.4, 4x Intel Arc Pro B70 32GB installed

## FP8 artifact status

- Official `Qwen/Qwen3.6-27B-FP8` is already downloaded locally as HF/Safetensors at `/home/steve/models/qwen3.6-27b-fp8-hf`.
- It is dynamic E4M3 block-FP8, not GGUF: `activation_scheme=dynamic`, `fmt=e4m3`, `weight_block_size=[128,128]`.
- I did not find a native FP8 GGUF for Qwen3.6 27B. Public repos with names like `Qwen3.6-27B-FP8-Q4_K_M-GGUF` expose Q4_K_M GGUF converted from an FP8 source, not native FP8 GGUF.
- Started downloading `vrfai/Qwen3.6-27B-FP8`, a compressed-tensors FP8 build with static/tensor activation and weight scales, to `/home/steve/models/qwen3.6-27b-fp8-vrfai/model.safetensors`.
- `hf download` repeatedly stalled/exited on the large file. A resumable curl loop works:

```bash
out=/home/steve/models/qwen3.6-27b-fp8-vrfai/model.safetensors
expected=35923194376
url='https://huggingface.co/vrfai/Qwen3.6-27B-FP8/resolve/main/model.safetensors?download=true'
while [ ! -f "$out" ] || [ "$(stat -c %s "$out")" -lt "$expected" ]; do
  curl -L --fail --silent --show-error --retry 20 --retry-all-errors --retry-delay 5 --continue-at - "$url" -o "$out" || true
  sleep 5
done
```

## vLLM/XPU MTP status

- vLLM has Qwen MTP wiring and the official Qwen3.6 FP8 checkpoint includes MTP weights.
- The current XPU Gated DeltaNet attention path blocks speculative decode: `/home/steve/src/vllm/vllm/_xpu_ops.py` asserts that `attn_metadata.spec_sequence_masks is None`.
- A possible correctness smoke patch is to make `GatedDeltaNetAttention.forward_xpu` fall back to the generic/CUDA-style path when speculative masks are present. This is not expected to be a speed path unless the downstream Triton/FLA kernels work well on XPU.

## Q8_1 activation cache prototype

Added an env-gated llama.cpp SYCL cache:

- Env: `GGML_SYCL_Q8_CACHE=1`.
- Scope: one graph compute.
- Key: source activation tensor, source data pointer, device, and Q8 shape.
- Purpose: reuse the exact Q8_1 activation for sibling Q4_0 matmuls and avoid repeated peer Q8 activation copies in tensor split.
- Quality impact: expected none. It reuses the same quantized activation the existing path already produced; it does not alter weights, sampling, KV dtype, or model math.

## Results

Q4_0 GGUF: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`.

| Mode | Selector | Output | Cache off | Cache on | Artifact |
| --- | --- | ---: | ---: | ---: | --- |
| 1x B70 | `level_zero:2` | 256 | 24.425 tok/s | 24.500 tok/s | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q8cache-single2-p512n256-20260504T153806Z-cache*.jsonl` |
| 2x B70 tensor | `level_zero:0,3` | 256 | 40.083 tok/s | 40.684 tok/s | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q8cache-dual03-p512n256-20260504T154039Z-cache*.jsonl` |
| 2x B70 tensor validation | `level_zero:0,3` | 512 | n/a | 40.487 tok/s | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q8cache-dual03-validate-p512n512-20260504T154332Z.jsonl` |
| 3x B70 tensor | `level_zero:2,1,3` | 256 | 40.937 tok/s | 42.432 tok/s | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q8cache-triple213-p512n256-20260504T154626Z-cache*.jsonl` |
| 3x B70 tensor validation | `level_zero:2,1,3` | 512 | n/a | 41.659 tok/s | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q8cache-triple213-validate-p512n512-20260504T154937Z.jsonl` |

3x cache result was submitted to LocalMaxxing with reduced payload due an API 500 on the full payload:

- Label: `llamacpp-qwen36-27b-q4_0-sycl-tp3-q8cache-root213-p512-n256-min`
- ID: `cmordq9t5000dl404x309pj48`
- tok/s out: `42.431805`
- tok/s total: `78.320035`

## Interpretation

Q8 caching is a modest multi-GPU optimization, not the single-card breakthrough. It helps where repeated activation quantization and peer Q8 copies are part of the tensor-split path. The larger remaining issue is still the high count of small per-token reductions and launch/synchronization overhead, especially for 4 GPUs.
