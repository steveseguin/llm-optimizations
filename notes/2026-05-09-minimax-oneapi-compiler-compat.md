# 2026-05-09 MiniMax oneAPI Compiler Compatibility

## Finding

The MiniMax AutoRound llm-scaler INT4 extension must currently be built with the
oneAPI 2025.3 compiler when used inside the active PyTorch XPU venv. Clean
builds with oneAPI 2026.0 produced a segmentation fault during Python import,
before any model tensors or vLLM code executed.

The crash occurred in SYCL offload image registration:

```text
sycl::_V1::detail::ProgramManager::addImage(...)
__sycl_register_lib
sycl.descriptor_reg
```

The active PyTorch XPU stack links `libsycl.so.8` from:

```text
/home/steve/.venvs/vllm-xpu/lib/libsycl.so.8
```

The system also has oneAPI 2026.0 installed, whose compiler/runtime is
`libsycl.so.9`. Rebuilding the extension with oneAPI 2026.0 generated a shared
object that linked against the venv `libsycl.so.8` but crashed during import.

## Working Build Command

```bash
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /home/steve/.venvs/vllm-xpu/bin/activate
cd /home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm
MAX_JOBS=2 TORCH_XPU_ARCH_LIST=bmg python setup_moe_int4_only.py build_ext --inplace -v
```

Import and standalone XPU kernel smoke passed after the 2025.3 rebuild:

```text
from custom_esimd_kernels_vllm import moe_forward_tiny_cutlass_nmajor_int4_u4
ok

torch.Size([1, 256]) torch.float16 True
```

MiniMax vLLM smoke and baseline also passed after the 2025.3 rebuild:

| run | prompt/output | output tok/s | total tok/s | notes |
| --- | ---: | ---: | ---: | --- |
| smoke | 1/8 | 3.8325 | 4.3110 | confirms model load and all 62 MoE layers enable |
| baseline | 512/512 | 36.0251 | 72.0502 | FP16 u4 path restored; near prior 37.1362 output tok/s |
| XPU graph requested | 512/512 | 29.5624 | 59.1249 | graph capture disabled for TP communication ops; negative result |
| CCL IPC pidfd | 512/512 | 35.5336 | 71.0672 | slightly slower than default IPC |
| max model len 1024 | 512/512 | 28.9094 | 57.8187 | smaller context budget; negative result |

## Impact

This unblocked the MiniMax AutoRound FP16 unsigned-u4 decode path after the
failed fused-router experiment. Treat oneAPI 2026.0 as unsafe for this extension
until the PyTorch XPU runtime moves to a compatible SYCL runtime or we verify a
specific build/link setup that imports cleanly.

## Logs

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-fp16-oneapi2025-20260509T201900Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-fp16-restore-20260509T201658Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-bf16-restore-with-compat-20260509T201430Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260509T202033Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T202757Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T203505Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T204253Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T205002Z.log
```

## TODO

- Rebuild the BF16 u4 patch with oneAPI 2025.3 and verify whether the previous
  BF16 path can be restored without the import crash.
- Keep fused router/top-2 work default-off until it can import and smoke-test
  cleanly.
