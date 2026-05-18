# MiniMax MoE Immediate Residual Allreduce Candidate

Status: rejected after full quality pass because performance was neutral/slower.

The tested source change did two things:

1. Treat `VLLM_MINIMAX_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE=1` like delayed MoE allreduce inside the MoE runner, preventing the runner's standalone final allreduce.
2. Immediately after `self.block_sparse_moe(hidden_states)` in `MiniMaxM2DecoderLayer.forward`, call `_allreduce_with_rank0_residual(hidden_states, residual)` and clear `residual`.

Tested conceptual diff:

```diff
diff --git a/vllm/model_executor/layers/fused_moe/runner/moe_runner.py b/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
@@
 def _minimax_moe_delay_allreduce_enabled() -> bool:
-    return os.environ.get("VLLM_MINIMAX_MOE_DELAY_ALLREDUCE", "0") == "1"
+    return (
+        os.environ.get("VLLM_MINIMAX_MOE_DELAY_ALLREDUCE", "0") == "1"
+        or os.environ.get("VLLM_MINIMAX_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE", "0")
+        == "1"
+    )
diff --git a/vllm/model_executor/models/minimax_m2.py b/vllm/model_executor/models/minimax_m2.py
@@
 _MINIMAX_M2_MOE_DELAY_ALLREDUCE = (
     os.environ.get("VLLM_MINIMAX_MOE_DELAY_ALLREDUCE", "0") == "1"
 )
+_MINIMAX_M2_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE = (
+    os.environ.get("VLLM_MINIMAX_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE", "0") == "1"
+)
@@
         hidden_states = self.block_sparse_moe(hidden_states)
+        if (
+            _MINIMAX_M2_MOE_IMMEDIATE_RESIDUAL_ALLREDUCE
+            and self.tp_size > 1
+            and residual is not None
+        ):
+            with timed_region("minimax.moe.immediate_residual_allreduce"):
+                hidden_states = self._allreduce_with_rank0_residual(
+                    hidden_states, residual
+                )
+            residual = None
         trace_tensor_graph(f"minimax.layer{self.layer_idx}.after_moe", hidden_states)
```

The patch was reverted from the active runtime after the benchmark. Keep this as a reference point: simple residual/allreduce relocation is quality-safe, but not faster than the promoted no-attention-delay logits-WS path.
