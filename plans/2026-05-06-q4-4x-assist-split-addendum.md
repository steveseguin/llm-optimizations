# 2026-05-06 Q4_0 Four-B70 Assist-Split Addendum

## Result

Four-card Qwen3.6 27B Q4_0 GGUF improved from the previous equal split `34.929313 tok/s` validation to `39.204149 tok/s` decode by treating the fourth B70 as a small assist device:

```text
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3,0
-dev SYCL0/SYCL1/SYCL2/SYCL3
-sm tensor -ts 1/1/1/0.05
GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2
```

This is quality preserving: same Q4_0 GGUF weights, f16 KV cache, no speculative decoding, and no power-limit change. LocalMaxxing accepted the result as `cmou581wv002dld0197mffpco`.

## Decision

Keep three-card Q4_0 tensor split as the best quality-preserving GGUF path for now. The four-card assist split is useful to document and may be useful when memory pressure or concurrent sessions matter, but it is not the current fastest single-session Q4_0 setup.

## Next Software Work

1. Investigate why narrow four-way Q4_0 reordered MMVQ shards lose efficiency.
2. Prioritize output-projection plus allreduce/residual epilogue fusion over more simple allreduce copy scheduling.
3. Use FP8/vLLM TP4 as the main all-four-card speed path while Q4_0 four-way kernel work continues.
4. Keep finer assist ratios such as `0.02`, `0.03`, `0.07`, and `0.12` as a low-cost follow-up, but only after higher-value kernel work or if a quick smoke window is available.

## Artifacts

- Note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-4x-assist-split.md`
- Data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-4x-assist-split-20260506.json`
- Validated JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-sg2-assist005-quad2130-p512n512-r3-20260506T141453Z.jsonl`
- LocalMaxxing response: `/home/steve/bench-results/localmaxxing-qwen36-q4-4x-assist005-retry-20260506T141904Z.json`
