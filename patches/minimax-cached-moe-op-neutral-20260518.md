# MiniMax Cached MoE Op Lookup Patch

Date: 2026-05-18

This patch was tested and rejected for promotion. It passed strict quality but did not beat the promoted baseline.

```diff
diff --git a/vllm/model_executor/layers/quantization/moe_wna16.py b/vllm/model_executor/layers/quantization/moe_wna16.py
--- a/vllm/model_executor/layers/quantization/moe_wna16.py
+++ b/vllm/model_executor/layers/quantization/moe_wna16.py
@@
     def __init__(self, quant_config: MoeWNA16Config, moe: "FusedMoEConfig") -> None:
         super().__init__(moe)
         self.quant_config = quant_config
+        self._llm_scaler_u4_op = None
+        self._llm_scaler_u4_ws_op = None
+        self._llm_scaler_u4_logits_op = None
+        self._llm_scaler_minimax_op = None
+        self._llm_scaler_minimax_ws_op = None
@@
+    def _get_llm_scaler_minimax_op(self, use_ws: bool):
+        if use_ws:
+            if self._llm_scaler_minimax_ws_op is None:
+                from custom_esimd_kernels_vllm import (
+                    moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws,
+                )
+                self._llm_scaler_minimax_ws_op = (
+                    moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws
+                )
+            return self._llm_scaler_minimax_ws_op
+        if self._llm_scaler_minimax_op is None:
+            from custom_esimd_kernels_vllm import (
+                moe_forward_tiny_cutlass_nmajor_int4_u4_minimax,
+            )
+            self._llm_scaler_minimax_op = (
+                moe_forward_tiny_cutlass_nmajor_int4_u4_minimax
+            )
+        return self._llm_scaler_minimax_op
@@
-                if self._llm_scaler_moe_minimax_logits_ws_requested():
-                    from custom_esimd_kernels_vllm import (
-                        moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws as moe_forward_tiny_cutlass_nmajor_int4_u4_minimax,
-                    )
+                use_minimax_ws = self._llm_scaler_moe_minimax_logits_ws_requested()
+                moe_forward_tiny_cutlass_nmajor_int4_u4_minimax = (
+                    self._get_llm_scaler_minimax_op(use_minimax_ws)
+                )
+                if use_minimax_ws:
                     logger.info_once(
                         "Using llm-scaler XPU INT4 MiniMax logits WS decode path"
                     )
```

The tested version also cached the generic U4 and U4 logits op callables. A one-shot optional contiguity logger was included for inspection, but it was left off during the benchmark.

Outcome:

- Quality: pass, with exact promoted token hashes across raw145 n64/n256, semantic, arithmetic-repeat, and extended sixpack.
- Speed: `82.006549` output tok/s, `109.342066` total tok/s mean.
- Baseline: `82.404268` output tok/s, `109.872357` total tok/s mean.
- Decision: rejected for promotion and reverted from active runtime. The Python dynamic import boundary is not the current decode bottleneck under the promoted graph recipe.
