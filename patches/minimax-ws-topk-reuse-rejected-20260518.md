# MiniMax WS TopK Reuse Patch

Date: 2026-05-18

This patch was tested and rejected. It failed the first raw exact gate with NUL/control-token corruption and was reverted from the active source.

```diff
diff --git a/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl b/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
--- a/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
+++ b/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
@@
+static bool reuse_minimax_ws_topk_buffers() {
+    static const bool value =
+        std::getenv("VLLM_XPU_MINIMAX_WS_REUSE_TOPK_ONLY") != nullptr;
+    return value;
+}
@@
-    const int n_tokens = (int)x.size(0);
-    auto topk_weight = torch::empty({n_tokens, top_k},
-        torch::device(router_logits.device()).dtype(torch::kFloat32));
-    auto topk_idx = torch::empty({n_tokens, top_k},
-        torch::device(router_logits.device()).dtype(torch::kInt32));
+    const int n_tokens = (int)x.size(0);
+    static thread_local torch::Tensor s_minimax_ws_topk_weight;
+    static thread_local torch::Tensor s_minimax_ws_topk_idx;
+    torch::Tensor topk_weight;
+    torch::Tensor topk_idx;
+    if (reuse_minimax_ws_topk_buffers()) {
+        const bool need_topk_weight =
+            !s_minimax_ws_topk_weight.defined() ||
+            s_minimax_ws_topk_weight.size(0) != n_tokens ||
+            s_minimax_ws_topk_weight.size(1) != top_k ||
+            s_minimax_ws_topk_weight.device() != router_logits.device();
+        const bool need_topk_idx =
+            !s_minimax_ws_topk_idx.defined() ||
+            s_minimax_ws_topk_idx.size(0) != n_tokens ||
+            s_minimax_ws_topk_idx.size(1) != top_k ||
+            s_minimax_ws_topk_idx.device() != router_logits.device();
+        if (need_topk_weight) {
+            s_minimax_ws_topk_weight = torch::empty({n_tokens, top_k},
+                torch::device(router_logits.device()).dtype(torch::kFloat32));
+        }
+        if (need_topk_idx) {
+            s_minimax_ws_topk_idx = torch::empty({n_tokens, top_k},
+                torch::device(router_logits.device()).dtype(torch::kInt32));
+        }
+        topk_weight = s_minimax_ws_topk_weight;
+        topk_idx = s_minimax_ws_topk_idx;
+    } else {
+        topk_weight = torch::empty({n_tokens, top_k},
+            torch::device(router_logits.device()).dtype(torch::kFloat32));
+        topk_idx = torch::empty({n_tokens, top_k},
+            torch::device(router_logits.device()).dtype(torch::kInt32));
+    }
```

Outcome:

- Raw145 n64 exact failed.
- Observed hash: `242152df6909e5e25433f43875de5e51c210d146a22279611852b695bcf7d978`
- Expected hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- NUL token count: `63`
- Decision: rejected and reverted.
