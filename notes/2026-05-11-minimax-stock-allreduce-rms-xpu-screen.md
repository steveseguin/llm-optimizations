# MiniMax Stock AllReduce RMS XPU Screen, 2026-05-11

## Goal

Check whether vLLM's existing `fuse_allreduce_rms` compiler pass can be reused
as a shortcut for MiniMax M2.7 AutoRound TP4 on the B70s. The target remains the
quality-cleared reference:

- p512/n1536 output: `37.552538` tok/s
- p512/n1536 total: `50.070051` tok/s
- LocalMaxxing: `cmozow03v005wlo01q81bnspx`
- guardrail: Q/K TP variance allreduce preserved; no speculation, expert
  dropping, or power-limit changes.

## Control

Before the pass screen, I reran a p512/n512 control on the current active vLLM
runtime in an isolated NVMe cache.

| Run | Shape | AOT cache | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | --- | --- | ---: | ---: | ---: | --- |
| current cold | p512/n512 | `9f17eb...` | 9,408 | `55.590941` | `27.795471` | cold AOT/KV artifact |
| current warm | p512/n512 | `9f17eb...` | 17,216 | `69.488127` | `34.744063` | below current anchors |

Logs:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-current-20260511/vllm-minimax-m27-autoround-tp4-p512n512-20260511T025849Z.log`
- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-current-20260511/vllm-minimax-m27-autoround-tp4-p512n512-20260511T030158Z.log`

The warm result is a valid control but not a LocalMaxxing result because it is
well below the prior valid p512/n512 and p512/n1536 MiniMax records.

## Patch Screen

I added a default-off XPU gate in the active vLLM source and installed runtime:

- `VLLM_XPU_EXPERIMENTAL_FUSE_ALLREDUCE_RMS=1` lets the XPU platform stop
  forcibly disabling `fuse_allreduce_rms`.
- `pass_manager.py` imports `AllReduceFusionPass` on XPU only under that env.

Patch artifact:

`patches/vllm-xpu-enable-stock-allreduce-rms-screen-20260511.patch`

This gate is intentionally not a runtime recommendation. It exists only to make
the failure mode reproducible without changing default B70 runs.

## Screens

All screens used TP4, vLLM/XPU 0.20.1-local, FP16 activations, AutoRound INT4
W4A16 weights, llm-scaler u4 decode enabled, and a tiny p64/n16 smoke.

| Screen | Extra config | Result |
| --- | --- | --- |
| first enable | `{"pass_config":{"fuse_allreduce_rms":true}}` | config validation calls CUDA capability lookup and fails with `Torch not compiled with CUDA enabled` |
| map max-size | `fi_allreduce_fusion_max_size_mb: {"4": 2}` | rejected by vLLM schema; current field expects a scalar float |
| scalar max-size | `fi_allreduce_fusion_max_size_mb: 2` | vLLM enables `allreduce_rms`, then worker startup imports FlashInfer CUDA code and fails |

The final scalar run reached:

```text
Enabled custom fusions: allreduce_rms
```

then each worker failed while importing:

```text
from .fusion.allreduce_rms_fusion import AllReduceFusionPass
import flashinfer.comm as _flashinfer_comm
...
torch.cuda.get_device_properties(0)
AssertionError: Torch not compiled with CUDA enabled
```

Logs:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-rms-pass-20260511/vllm-minimax-m27-autoround-tp4-p64n16-20260511T030636Z.log`
- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-rms-pass-20260511/vllm-minimax-m27-autoround-tp4-p64n16-20260511T030722Z.log`
- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ar-rms-pass-20260511/vllm-minimax-m27-autoround-tp4-p64n16-20260511T031029Z.log`

## Decision

Do not use `VLLM_XPU_EXPERIMENTAL_FUSE_ALLREDUCE_RMS=1` for real benchmarks.
The stock vLLM allreduce/RMS pass is tied to FlashInfer CUDA import paths and is
not a portable B70 shortcut.

The next implementation path remains XPU-specific:

1. Keep the allreduce/RMS target, but avoid FlashInfer and CUDA imports.
2. Start by improving generated-graph inspection for the current cache layout so
   we can count allreduce/wait/RMS boundaries reliably after every compile.
3. Prototype a narrower XPU compiler transformation or lowered op that preserves
   `_c10d_functional.all_reduce_` semantics while reducing the post-wait
   scheduler boundary.

Structured data:

`data/minimax-m27-stock-allreduce-rms-xpu-screen-20260511.json`
