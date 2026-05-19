# Patch Tried: MiniMax Attention Post-Reduce Add+RMS XPU Helper

This candidate added a default-off `VLLM_MINIMAX_ATTN_POST_REDUCE_RMS_XPU=1`
hook that swapped MiniMax's post-attention `RMSNorm(hidden_states, residual)`
call for an XPU `add_rms` helper after the normal attention output allreduce.

The helper microcheck was exact, but the integrated model failed `raw145-n64`
and decoded at only about `9.52 tok/s`, so the active vLLM hook was removed.

```diff
diff --git a/experiments/minimax_ar_fused_rms_xpu/__init__.py b/experiments/minimax_ar_fused_rms_xpu/__init__.py
@@
 @torch.library.register_fake("minimax_ar_fused_rms_xpu::ar_rms")
 def _fake_ar_rms(input, weight, group_name: str, eps: float):
     return torch.empty_like(input), torch.empty_like(input)
+
+
+@torch.library.register_fake("minimax_ar_fused_rms_xpu::add_rms")
+def _fake_add_rms(input, residual, weight, eps: float):
+    return torch.empty_like(input), torch.empty_like(input)

diff --git a/experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp b/experiments/minimax_ar_fused_rms_xpu/minimax_ar_fused_rms_xpu.cpp
@@
+std::tuple<at::Tensor, at::Tensor> add_rms(const at::Tensor& input,
+                                           const at::Tensor& residual,
+                                           const at::Tensor& weight,
+                                           double eps) {
+  const at::DeviceGuard device_guard(input.device());
+  CHECK_XPU(input);
+  CHECK_XPU(residual);
+  CHECK_XPU(weight);
+  CHECK_CONTIGUOUS(input);
+  CHECK_CONTIGUOUS(residual);
+  CHECK_CONTIGUOUS(weight);
+  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
+  TORCH_CHECK(residual.sizes() == input.sizes(), "residual shape mismatch");
+  TORCH_CHECK(weight.dim() == 1, "weight must be rank-1");
+  TORCH_CHECK(weight.size(0) == input.size(1), "weight hidden size mismatch");
+  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
+              "input and residual dtype mismatch");
+  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
+              "input and weight dtype mismatch");
+
+  at::Tensor out = at::empty_like(input);
+  at::Tensor residual_out = at::empty_like(input);
+
+  AT_DISPATCH_REDUCED_FLOATING_TYPES(
+      input.scalar_type(), "minimax_add_rms_xpu", [&] {
+        launch_ar_fused_add_rms<scalar_t>(input,
+                                          residual,
+                                          weight,
+                                          out,
+                                          residual_out,
+                                          static_cast<float>(eps));
+      });
+  return std::make_tuple(out, residual_out);
+}
+ 
 TORCH_LIBRARY(minimax_ar_fused_rms_xpu, m) {
@@
+  m.def(
+      "add_rms(Tensor input, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
 }
 
 TORCH_LIBRARY_IMPL(minimax_ar_fused_rms_xpu, XPU, m) {
@@
+  m.impl("add_rms", add_rms);
 }

diff --git a/vllm/model_executor/models/minimax_m2.py b/vllm/model_executor/models/minimax_m2.py
@@
+_MINIMAX_ATTN_POST_REDUCE_RMS_XPU = (
+    os.environ.get("VLLM_MINIMAX_ATTN_POST_REDUCE_RMS_XPU", "0") == "1"
+)
@@
-if _MINIMAX_AR_FUSED_RMS_XPU or _MINIMAX_AR_RMS_XPU:
+if (
+    _MINIMAX_AR_FUSED_RMS_XPU
+    or _MINIMAX_AR_RMS_XPU
+    or _MINIMAX_ATTN_POST_REDUCE_RMS_XPU
+):
@@
+_MINIMAX_ATTN_POST_REDUCE_RMS_XPU_AVAILABLE = (
+    _MINIMAX_ATTN_POST_REDUCE_RMS_XPU
+    and _minimax_ar_fused_rms_xpu is not None
+)
@@
+        elif _MINIMAX_ATTN_POST_REDUCE_RMS_XPU_AVAILABLE:
+            with timed_region("minimax.attn.post_reduce_add_rms_xpu"):
+                hidden_states, residual = (
+                    torch.ops.minimax_ar_fused_rms_xpu.add_rms(
+                        hidden_states.contiguous(),
+                        residual.contiguous(),
+                        self.post_attention_layernorm.weight.data,
+                        self.post_attention_layernorm.variance_epsilon,
+                    )
+                )
```
