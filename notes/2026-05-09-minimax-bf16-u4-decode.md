# 2026-05-09 MiniMax BF16 U4 Decode

## Result

The MiniMax AutoRound llm-scaler u4 decode path now supports BF16 hidden states. This fixes the earlier BF16 fallback behavior without forcing the model through FP16 activations.

| run | prompt/output | output tok/s | total tok/s | notes |
| --- | ---: | ---: | ---: | --- |
| BF16 before scale-gate fix | 512/256 | 16.860287 | 50.580862 | no llm-scaler MoE enable logs; vLLM fallback path |
| BF16 u4 decode path | 512/256 | 33.681326 | 101.043979 | all 62 MoE layers enabled, native BF16 hidden states |
| BF16 u4 decode path | 512/512 | 36.607699 | 73.215399 | steady decode validation |
| FP16 u4 decode reference | 512/512 | 37.136187 | 74.272373 | current fastest AutoRound run |

The BF16 p512/n512 result is only about 1.4% slower than the FP16 p512/n512 reference, while avoiding the activation dtype change. Compared with the pre-fix BF16 fallback at p512/n256, the patched path is about 2.0x faster.

LocalMaxxing accepted the BF16 p512/n512 result as `cmoyr84ol000rtl01o4z9fwdm`.

## Commands

```bash
USE_LLM_SCALER_MOE=1 CCL_IPC=default XPU_GRAPH=0 DTYPE=bfloat16 \
INPUT_LEN=512 OUTPUT_LEN=256 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 \
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh

USE_LLM_SCALER_MOE=1 CCL_IPC=default XPU_GRAPH=0 DTYPE=bfloat16 \
INPUT_LEN=512 OUTPUT_LEN=512 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 \
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

## Logs

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-bf16-20260509T191519Z.log

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T191758Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T191758Z.json

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T192643Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T192643Z.json

/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T193458Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T193458Z.json

/home/steve/bench-results/localmaxxing-minimax-m27-autoround-u4-decode-bf16-p512n512-20260509.response.json
```

The good BF16 logs show all 62 MiniMax MoE layers enabled:

```text
Enabled llm-scaler XPU INT4 MoE decode path for model.layers.0.mlp.experts (scale_dtype=torch.bfloat16)
...
Enabled llm-scaler XPU INT4 MoE decode path for model.layers.61.mlp.experts (scale_dtype=torch.bfloat16)
```

## What Changed

The llm-scaler scalar u4 tiny-MoE up/down kernels were templated over the hidden-state dtype and now accept both FP16 and BF16 inputs/outputs. Accumulation remains FP32, and the packed AutoRound u4 nibble decode remains `nibble - 8`.

The vLLM WNA16 MiniMax hook now:

- allows `torch.bfloat16` decode activations in addition to `torch.float16`;
- accepts BF16 checkpoint scale tensors at load time;
- materializes FP16 scale copies for the llm-scaler kernels when the checkpoint scales are BF16;
- keeps the custom path restricted to decode-sized batches (`x.shape[0] <= 4`).

Standalone BF16 kernel validation on a fake MiniMax-shaped routed layer produced finite BF16 output. A small reference check against a torch-dequantized u4 implementation had `max_abs=0.001461`, `mean_abs=0.000190`, and `mean_rel=0.005335`.

## Patches

```text
patches/llm-scaler-moe-int4-u4-bf16-decode-20260509.patch
patches/vllm-minimax-llm-scaler-u4-bf16-decode-20260509.patch
```

The active site package copy of `vllm/model_executor/layers/quantization/moe_wna16.py` is identical to the source-tree copy after this patch.

## Interpretation

This is useful because it improves quality-preserving operation rather than chasing speed through a more aggressive runtime dtype change. The result does not beat the fastest FP16 AutoRound number, but it closes the BF16 performance gap almost completely. The next speed work should stay focused on per-layer overhead outside the MoE matvec itself: router/top-k glue, attention/KV update cost, and MiniMax Q/K RMS plus tensor-parallel collectives.
