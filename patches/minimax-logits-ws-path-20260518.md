# MiniMax Logits-To-WS Path Patch Notes

Date: 2026-05-18

This patch is layered on top of the existing llm-scaler unsigned-u4 MiniMax work-sharing MoE path. It is not a standalone upstream patch from vanilla vLLM; apply after the previously recorded work-sharing INT4 MoE patches.

## Source Files

- `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`
- `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/ops.py`
- `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/__init__.py`
- `/home/steve/src/vllm/vllm/model_executor/layers/quantization/moe_wna16.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/moe_wna16.py`
- `/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh`

## Functional Delta

- Add native op `moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws`.
- The op computes exact MiniMax M2 top-8 routing from `router_logits` using the existing sigmoid plus `e_score_bias` host helper.
- The op calls `moe_forward_tiny_cutlass_nmajor_int4_ws_impl(..., signed_compact=false)` so unsigned-u4 AutoRound weights keep the faster work-sharing MoE decode path.
- Add Python wrapper and package export.
- Add vLLM env selector `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`.
- Require the runtime log marker `Using llm-scaler XPU INT4 MiniMax logits WS decode path` in the strict benchmark run.

## Build

```bash
set +u
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
set -u
source /home/steve/.venvs/vllm-xpu/bin/activate
export LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib:${LD_LIBRARY_PATH:-}
cd /home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm
python setup_moe_int4_only.py build_ext --inplace
```

## Smoke Test

```bash
PYTHONPATH=/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python \
LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib:${LD_LIBRARY_PATH:-} \
/home/steve/.venvs/vllm-xpu/bin/python - <<'PY'
from custom_esimd_kernels_vllm import moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws
import torch
import custom_esimd_kernels_vllm.moe_int4_ops
print("wrapper_ok", callable(moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws))
print("torch_op_ok", hasattr(torch.ops.moe_int4_ops, "moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws"))
PY
```

Expected:

```text
wrapper_ok True
torch_op_ok True
```

## Validation Result

Full strict quality passed and the p512/n1536 two-repeat benchmark averaged `81.758267` output tok/s and `109.011023` total tok/s. See `notes/2026-05-18-minimax-logits-ws-strict-win.md`.
