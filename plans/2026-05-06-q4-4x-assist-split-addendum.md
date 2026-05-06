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
2. Keep the current single-kernel allreduce-add path for validated runs; the follow-up communication sweep did not find a better existing flag combination.
3. Use FP8/vLLM TP4 as the main all-four-card speed path while Q4_0 four-way kernel work continues.
4. Build and test isolated MMV_Y / row-grouping variants before changing the known-good binary.

## Follow-Up Probes

- Fine assist sweep: `/home/steve/bench-results/qwen36-q4_0-gguf/tensorsplit-quad-sg2-fine-assist-ratio-p0n128-r2-20260506T143835Z.tsv`
- Result: ratios `0.03` through `0.12` stayed below the validated `39.204149 tok/s`; ratios `0.01` and `0.02` failed with Level Zero out-of-device-memory during `MUL_MAT`.
- Allreduce stats: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-stats4-assist005-quad2130-p0n1-r1-20260506T145113Z.log`
- Result: `128` allreduces per generated token, `20,480` bytes each, warm total `4.213 ms/token`.
- Fused2 debug: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-fused2-triple213-p0n1-r1-20260506T144944Z.log`
- Result: fused2 is already active in tensor split, so the next Q4 source work is not simple fused2 enablement.
- Communication flags: `/home/steve/bench-results/qwen36-q4_0-gguf/comm-flag-sweep-quad-assist005-p0n128-r2-20260506T145544Z.tsv`
- Result: no existing communication flag materially beat baseline; pairwise, striped, and no-fuseadd small-f32 were clearly slower.
- MMV_Y=2 probe: `/home/steve/bench-results/qwen36-q4_0-gguf/mmv-y2-probe-p0n128-r2-20260506T152341Z.tsv`
- Result: neutral on 3x and not useful for 4x assist.
- MUL_MAT stage stats: `/home/steve/bench-results/qwen36-q4_0-gguf/mulmat-stage-current-3x-vs-4x-p0n1-r1-20260506T152722Z.tsv`
- Result: 4x assist adds `282` quantization calls and `538` matmul kernel launches versus 3x for essentially the same matmul byte volume.
- Skip-last threshold patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-split-skip-last-below-rows-focused-20260506.patch`
- Result: safe when unset, but threshold sweeps did not beat the validated 4x assist command. Keep `GGML_SYCL_SPLIT_SKIP_LAST_BELOW_ROWS` unset for production.

## Artifacts

- Note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-4x-assist-split.md`
- Follow-up note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-narrow-shard-followups.md`
- Data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-4x-assist-split-20260506.json`
- Follow-up data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-narrow-shard-followups-20260506.json`
- Validated JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-sg2-assist005-quad2130-p512n512-r3-20260506T141453Z.jsonl`
- LocalMaxxing response: `/home/steve/bench-results/localmaxxing-qwen36-q4-4x-assist005-retry-20260506T141904Z.json`
