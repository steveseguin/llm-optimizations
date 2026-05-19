# Rejected Patch: INC Fast 2D Apply

Date: 2026-05-19

This patch was quality-safe but slower than the current clean MiniMax high, so it was removed from the active source and venv copies after testing.

```diff
diff --git a/vllm/model_executor/layers/quantization/inc.py b/vllm/model_executor/layers/quantization/inc.py
--- a/vllm/model_executor/layers/quantization/inc.py
+++ b/vllm/model_executor/layers/quantization/inc.py
@@
-from fractions import Fraction
+import os
+from fractions import Fraction
@@
 logger = init_logger(__name__)
+
+_INC_XPU_FAST_2D_APPLY = os.environ.get("VLLM_XPU_INC_FAST_2D_APPLY", "0") == "1"
@@
     ) -> torch.Tensor:
         # qweight is already in NT layout [K_packed, N] (strides (1, K_packed))
         # from process_weights_after_loading — pass directly to kernel.
+        if _INC_XPU_FAST_2D_APPLY and x.dim() == 2 and x.is_contiguous():
+            return torch.ops._xpu_C.int4_gemm_w4a16(
+                x,
+                layer.qweight,
+                bias,
+                layer.scales,
+                layer.qzeros,
+                self.group_size,
+                None,  # g_idx not needed: desc_act is always False for INC models
+            )
         out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
         reshaped_x = x.reshape(-1, x.shape[-1])
         out = torch.ops._xpu_C.int4_gemm_w4a16(
```
