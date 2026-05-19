# Patch Tried: MiniMax Greedy Sampler FP32-Skip Current Baseline

This patch was tested with `VLLM_XPU_GREEDY_SKIP_LOGITS_FP32=1` and rejected
because `raw145-n256-exact` failed the expected combined token hash on the
current clean MiniMax baseline.

```diff
diff --git a/vllm/v1/sample/sampler.py b/vllm/v1/sample/sampler.py
index a77eafba2..b4ae0b784 100644
--- a/vllm/v1/sample/sampler.py
+++ b/vllm/v1/sample/sampler.py
@@ -2,6 +2,8 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """A layer that samples the next tokens from the model's outputs."""
 
+import os
+
 import torch
 import torch.nn as nn
 
@@ -87,6 +89,24 @@ class Sampler(nn.Module):
                 else:
                     raw_logprobs = logits.to(torch.float32)
 
+        if (
+            os.environ.get("VLLM_XPU_GREEDY_SKIP_LOGITS_FP32", "0") == "1"
+            and logits.device.type == "xpu"
+            and sampling_metadata.all_greedy
+            and num_logprobs is None
+            and not sampling_metadata.logprob_token_ids
+            and sampling_metadata.no_penalties
+            and sampling_metadata.allowed_token_ids_mask is None
+            and not sampling_metadata.bad_words_token_ids
+            and not sampling_metadata.logitsprocs.argmax_invariant
+            and not sampling_metadata.logitsprocs.non_argmax_invariant
+        ):
+            sampled = self.greedy_sample(logits).to(torch.int32)
+            return SamplerOutput(
+                sampled_token_ids=sampled.unsqueeze(-1),
+                logprobs_tensors=None,
+            )
+
         # Use float32 for the logits.
         logits = logits.to(torch.float32)
 
```
