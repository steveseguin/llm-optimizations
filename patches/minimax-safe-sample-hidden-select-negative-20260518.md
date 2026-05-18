# MiniMax Safe Sample Hidden Select Negative Patch Note

Date: 2026-05-18

No new runtime patch was produced for this candidate. It tested the existing guarded runtime path in `gpu_model_runner.py`:

```bash
VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1
```

The candidate passed the full strict quality gate but was slower than the promoted logits-WS baseline:

- Candidate mean: `77.314354` output tok/s, `103.085805` total tok/s
- Promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s

Harness metadata fix applied after the run:

```diff
+    --arg vllm_xpu_safe_sample_hidden_select "${VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT:-}" \
+        VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT: $vllm_xpu_safe_sample_hidden_select,
```

Do not promote this flag for the current MiniMax M2.7 AutoRound TP4 recipe.
