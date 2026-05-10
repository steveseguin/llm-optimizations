# MiniMax Source-Tree IR Fused-Add RMS Screen, 2026-05-10

## Goal

Test whether the newer source-tree vLLM IR path for `fused_add_rms_norm` can
use the existing XPU fused add+RMSNorm kernel to reduce MiniMax per-layer
residual/RMS overhead.

This is quality-preserving: it changes only the implementation of residual-add
plus RMSNorm, not model weights, quantization, routing, Q/K TP variance
allreduce, speculation, or GPU power.

## Why This Was Tested

The installed runtime exposes only `rms_norm` in `IrOpPriorityConfig`, while
`/home/steve/src/vllm` has a newer `fused_add_rms_norm` IR op and an XPU
implementation. That makes source import a convenient way to test the boundary:

```bash
LLM_SCALER_KERNELS=/home/steve/src/vllm:/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python
EXTRA_ARGS='--ir-op-priority {"fused_add_rms_norm":["xpu_kernels"]}'
```

## Source-Tree Setup Fix

The first source-tree run fell back to the default MoE config because the B70
MiniMax config existed in site-packages but not in `/home/steve/src/vllm`.
I added the same config to the source tree:

```text
/home/steve/src/vllm/vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json
```

Patch artifact:

```text
patches/vllm-source-b70-minimax-moe-config-20260510.patch
```

## Results

All runs are TP4, FP16, `USE_LLM_SCALER_MOE=1`, p512/n512,
`max_model_len=2048`, `max_num_batched_tokens=512`, no speculation, no power
change.

| Label | Runtime | Key config | KV tokens | Total tok/s | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T160342Z` | installed | `custom_ops=["none","+rms_norm"]`, cold | 9,472 | 57.245 | 28.622 | cold artifact |
| `20260510T160716Z` | installed | `custom_ops=["none","+rms_norm"]`, warm | 17,920 | 72.317 | 36.159 | negative vs installed baseline |
| `20260510T161048Z` | source | fused-add XPU, missing B70 MoE config, cold | 9,472 | 55.951 | 27.976 | invalid setup |
| `20260510T161400Z` | source | fused-add XPU, missing B70 MoE config, warm | 17,920 | 70.315 | 35.158 | invalid setup |
| `20260510T161658Z` | source | fused-add XPU, B70 MoE config present, warm | 17,920 | 71.298 | 35.649 | negative |
| `20260510T161935Z` | source | default IR, B70 MoE config present, cold | 9,472 | 55.701 | 27.850 | cold artifact |
| `20260510T162245Z` | source | default IR, B70 MoE config present, warm | 17,920 | 69.204 | 34.602 | source control |
| `20260510T162618Z` | source | default IR, forced FlashInfer autotune, cold | 9,472 | 55.941 | 27.971 | cold artifact |
| `20260510T162933Z` | source | default IR, forced FlashInfer autotune, warm | 17,920 | 71.562 | 35.781 | negative vs installed baseline |

Reference comparison:

- installed-runtime accepted p512/n512 reference: `39.610585` output tok/s
- installed-runtime p512/n1536 quality-conservative anchor: `37.552538` output tok/s

## Interpretation

The source-tree `fused_add_rms_norm=["xpu_kernels","native"]` route is viable
and mechanically applies, but it does not beat the installed runtime. Within the
source tree it is roughly neutral to slightly positive versus source default
after the B70 MoE config is present, but source import itself is several tok/s
slower than the installed package.

The installed-runtime `custom_ops=["none","+rms_norm"]` screen is also negative
at p512/n512. It reaches only `36.159` output tok/s after warm AOT reuse.

Do not submit these to LocalMaxxing. They are implementation screens below the
current valid reference.

## Follow-up

- Keep the B70 MiniMax MoE config in the source tree for future source-import
  experiments.
- Do not use source-tree import for official MiniMax results unless its baseline
  first matches the installed-runtime floor.
- If revisiting this path, backport only the minimal installed-runtime IR
  fused-add pieces behind a default-off flag, then compare against installed
  p512/n512 and p512/n1536 references.
- The main 60 tok/s path is still deeper fusion around TP collectives and
  adjacent epilogues, not simply swapping RMSNorm providers.
