# MiniMax Graph-Visible Clone Custom Allreduce Hang

Date: 2026-05-18 local / 2026-05-19 UTC

## Summary

Tested a graph-visible clone variant of the clone-safe compiled custom allreduce path:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1`
- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `CCL_TOPO_P2P_ACCESS=1`

The idea was to preserve the faster `torch.ops.vllm.all_reduce` custom-op path while moving the alias-safety clone into the compiled graph, instead of cloning inside the Python custom allreduce implementation.

## Outcome

The candidate is rejected.

The raw145 n64 smoke test loaded the model, enabled the llm-scaler INT4 MoE decode path for all MiniMax layers, and completed AOT graph compilation:

- compile range `(1, 1)`: `76.28 s`
- compile range `(1, 512)`: `65.31 s`
- torch.compile total: `171.40 s`

After compilation it repeatedly emitted shared-memory broadcast wait warnings and never wrote the raw145 n64 quality JSON. The run was terminated manually before the 30-minute timeout because this was already outside the normal post-compile window for a 64-token canary.

## Decision

Do not promote and do not benchmark. This candidate failed before producing the first exact-token canary, so it is not eligible for LocalMaxxing.

The failure suggests that placing the clone inside the captured graph changes graph/comm replay behavior enough to deadlock or stall the worker/broadcast path on this XPU setup. Keep the promoted clone-safe custom allreduce recipe, where `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1` performs the clone inside the custom allreduce implementation.

## Artifacts

- Smoke directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/graph-clone-custom-allreduce-smoke-20260519T000420Z`
- Smoke log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/graph-clone-custom-allreduce-smoke-20260519T000420Z/raw145-n64.log`
