# MiniMax Q/K Fusion Pass Retest, 2026-05-11

## Why

The external recipe suggested enabling vLLM's MiniMax Q/K norm fusion:

```bash
--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'
```

This is worth tracking because vLLM documents MiniMax-specific Q/K norm fusion,
but upstream's fast path is not directly available on XPU. The local XPU helper
path is default-off behind `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1`.

## Retest

Command shape:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- TP: `4`
- dtype: `float16`
- input/output: p64/n128
- `MAX_MODEL_LEN=512`
- `MAX_BATCHED_TOKENS=256`
- `USE_LLM_SCALER_MOE=1`
- `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1`
- `LD_LIBRARY_PATH` included `/opt/intel/oneapi/compiler/2025.3/lib`
- isolated cache root:
  `/mnt/fast-ai/vllm-cache/minimax-qkpass-20260511T005639Z`

Result:

- total: `25.238492` tok/s
- output-equivalent: `16.826` tok/s
- elapsed: `7.607427` s
- GPU KV cache size: `9,472` tokens
- compile time: `32.43` s
- log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qkpass-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T005639Z.log`
- json:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qkpass-20260511/vllm-minimax-m27-autoround-tp4-p64n128-20260511T005639Z.json`

The isolated cache produced the familiar low-KV cold compile artifact, so this
single p64/n128 number is not the only reason to reject the pass. More important
is that it matches the prior 2026-05-10 Q/K helper-pass conclusion: even a warm
p512/n1536 run reached only `37.24` output tok/s, below the accepted
quality-cleared anchor at `37.552538`.

## Decision

Keep `fuse_minimax_qk_norm` closed for the current XPU helper implementation.
The helper can reduce some local Q/K arithmetic, but it does not remove the
real scheduling/collective boundary and can perturb the compiled graph enough
to lose the existing INT4/RMS fusion shape.

Do not submit this run to LocalMaxxing. Future Q/K work should be a real XPU
collective-plus-RMS kernel or compiler lowering that preserves the existing
compiled schedule, not another helper-backed Inductor pattern swap.

