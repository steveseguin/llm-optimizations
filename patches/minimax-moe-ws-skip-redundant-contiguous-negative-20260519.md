# MiniMax MoE WS Skip Redundant Contiguous Patch

Date: 2026-05-19

Result: quality-safe, speed-negative. Do not promote.

## Source Patch

File: `vllm/model_executor/layers/quantization/moe_wna16.py`

```diff
@@
 _CACHE_MINIMAX_LOGITS_OP = (
     os.environ.get("VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP", "0") == "1"
 )
+_MINIMAX_LOGITS_SKIP_REDUNDANT_CONTIGUOUS = (
+    os.environ.get(
+        "VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS", "0"
+    )
+    == "1"
+)
@@
-                return moe_forward_tiny_cutlass_nmajor_int4_u4_minimax(
-                    x.contiguous(),
+                x_for_moe = (
+                    x
+                    if _MINIMAX_LOGITS_SKIP_REDUNDANT_CONTIGUOUS
+                    and x.is_contiguous()
+                    else x.contiguous()
+                )
+                router_logits_for_moe = (
+                    router_logits
+                    if _MINIMAX_LOGITS_SKIP_REDUNDANT_CONTIGUOUS
+                    and router_logits.is_contiguous()
+                    else router_logits.contiguous()
+                )
+
+                return moe_forward_tiny_cutlass_nmajor_int4_u4_minimax(
+                    x_for_moe,
                     layer.w13_qweight,
                     layer._llm_scaler_w13_scales,
                     layer.w2_qweight,
                     layer._llm_scaler_w2_scales,
-                    router_logits.contiguous(),
+                    router_logits_for_moe,
                     layer.e_score_correction_bias,
                     int(layer.top_k),
                     bool(layer.renormalize),
```

## Harness Capture

File: `scripts/run-minimax-strict-quality-gated-candidate.sh`

```diff
@@
+    --arg vllm_xpu_llm_scaler_moe_minimax_skip_redundant_contiguous "${VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS:-}" \
@@
+        VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS: $vllm_xpu_llm_scaler_moe_minimax_skip_redundant_contiguous,
```

## Reproduction Command

```bash
LABEL=minimax-moe-ws-skip-redundant-contiguous-20260519 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=4 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-moe-ws-skip-redundant-contiguous-20260519 \
VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1 \
VLLM_XPU_LLM_SCALER_MOE_MINIMAX_SKIP_REDUNDANT_CONTIGUOUS=1 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

Important inherited current-best envs included `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`,
`VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`,
`VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`,
`VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`,
`VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2`,
`VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0`,
`VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0`,
`VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`,
`VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4`,
`VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1`, and
`VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`.

## Outcome

- Quality: passed raw145 n64/n256 exact, semantic suite, 16-repeat arithmetic, and extended sixpack.
- Mean output: `88.885135` tok/s.
- Mean total: `118.513514` tok/s.
- Decision: reject; below the current promoted `88.927945` output tok/s strict high.
