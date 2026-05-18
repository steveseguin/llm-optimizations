# MiniMax Safe Sample Hidden Select Neutral Patch Note

Date: 2026-05-18

No new runtime patch was produced for this candidate. It tested the existing guarded runtime path in `gpu_model_runner.py`:

```bash
VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1
```

Harness fixes applied while recording this result:

1. Include the safe-selector flag in strict-runner candidate summaries:

```diff
+    --arg vllm_xpu_safe_sample_hidden_select "${VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT:-}" \
+        VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT: $vllm_xpu_safe_sample_hidden_select,
```

2. Change the strict runner default attention backend from `TRITON_ATTN` to `default`, matching the promoted MiniMax FlashAttention baseline:

```diff
-ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
+ATTENTION_BACKEND="${ATTENTION_BACKEND:-default}"
```

This second fix matters for reproducibility. The first safe-selector result used the older `TRITON_ATTN` default and produced `77.314354` output tok/s. That run passed quality, but it was backend-mismatched and should be treated as diagnostic only.

Fair FlashAttention rerun:

- Candidate mean: `81.914167` output tok/s, `109.218890` total tok/s
- Promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s
- Delta: `+0.19%` output tok/s, within normal run variance

Decision: quality-clean neutral/tie. Do not promote this flag or submit it to LocalMaxxing for the current MiniMax M2.7 AutoRound TP4 recipe.
