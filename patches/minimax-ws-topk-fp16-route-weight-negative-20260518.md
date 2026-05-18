# MiniMax WS TopK FP16 Route-Weight Patch

Date: 2026-05-18

This patch was tested and rejected for performance. It passed strict quality but did not beat the promoted baseline.

The local llm-scaler source tree already contains other MiniMax work-sharing changes, so this is recorded as the incremental idea rather than a clean repo-wide diff.

```diff
diff --git a/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl b/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
--- a/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
+++ b/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl
@@
+static bool minimax_ws_topk_weight_fp16() {
+    static const bool value =
+        std::getenv("VLLM_XPU_MINIMAX_WS_TOPK_WEIGHT_FP16") != nullptr;
+    return value;
+}
@@
-template<int NUM_EXPERTS, int TOPK, typename LogitsT>
+template<int NUM_EXPERTS, int TOPK, typename LogitsT, typename WeightT>
 struct MiniMaxM2TopKSigmoidBiasKernel {
     const LogitsT* router_logits;
     const float* e_score_bias;
-    float* top_values;
+    WeightT* top_values;
     int32_t* top_indices;
@@
-        float* vp = top_values + (size_t)row * TOPK;
+        WeightT* vp = top_values + (size_t)row * TOPK;
@@
-            block_store<float, 8>(vp, sv8);
+            if constexpr (std::is_same_v<WeightT, float>) {
+                block_store<float, 8>(vp, sv8);
+            } else {
+                block_store<WeightT, 8>(vp, convert<WeightT>(sv8));
+            }
@@
-template<typename LogitsT>
+template<typename LogitsT, typename WeightT>
 static void minimax_m2_top8_sigmoid_bias_host(
     const LogitsT* router_logits,
     const float* e_score_bias,
-    float* top_values,
+    WeightT* top_values,
@@
-            MiniMaxM2TopKSigmoidBiasKernel<256, 8, LogitsT>{
+            MiniMaxM2TopKSigmoidBiasKernel<256, 8, LogitsT, WeightT>{
                 router_logits, e_score_bias, top_values, top_indices, n_tokens, norm});
@@
-    auto topk_weight = torch::empty({n_tokens, top_k},
-        torch::device(router_logits.device()).dtype(torch::kFloat32));
     auto topk_idx = torch::empty({n_tokens, top_k},
         torch::device(router_logits.device()).dtype(torch::kInt32));
@@
-    if (router_logits.scalar_type() == torch::kFloat) {
-        minimax_m2_top8_sigmoid_bias_host<float>(
-            router_logits.data_ptr<float>(), e_score_bias.data_ptr<float>(),
-            topk_weight.data_ptr<float>(), topk_idx.data_ptr<int32_t>(),
-            n_tokens, norm, queue);
+    if (minimax_ws_topk_weight_fp16()) {
+        auto topk_weight = torch::empty({n_tokens, top_k},
+            torch::device(router_logits.device()).dtype(torch::kHalf));
+        if (router_logits.scalar_type() == torch::kFloat) {
+            minimax_m2_top8_sigmoid_bias_host<float>(
+                router_logits.data_ptr<float>(), e_score_bias.data_ptr<float>(),
+                (fp16*)topk_weight.data_ptr(), topk_idx.data_ptr<int32_t>(),
+                n_tokens, norm, queue);
+        } else {
+            minimax_m2_top8_sigmoid_bias_host<fp16>(
+                (const fp16*)router_logits.data_ptr(), e_score_bias.data_ptr<float>(),
+                (fp16*)topk_weight.data_ptr(), topk_idx.data_ptr<int32_t>(),
+                n_tokens, norm, queue);
+        }
+        return moe_forward_tiny_cutlass_nmajor_int4_ws_impl(
+            x, w13_qweight_u4, w13_scales, w2_qweight_u4, w2_scales,
+            topk_weight, topk_idx, false);
     } else {
-        minimax_m2_top8_sigmoid_bias_host<fp16>(
-            (const fp16*)router_logits.data_ptr(), e_score_bias.data_ptr<float>(),
-            topk_weight.data_ptr<float>(), topk_idx.data_ptr<int32_t>(),
-            n_tokens, norm, queue);
+        auto topk_weight = torch::empty({n_tokens, top_k},
+            torch::device(router_logits.device()).dtype(torch::kFloat32));
+        ...
     }
```

Outcome:

- Quality: pass, with exact promoted token hashes across raw145 n64/n256, semantic, arithmetic-repeat, and extended sixpack.
- Speed: `81.417643` output tok/s, `108.556857` total tok/s mean.
- Baseline: `82.404268` output tok/s, `109.872357` total tok/s mean.
- Decision: rejected for performance. Keep FP32 route weights in the promoted runtime.
