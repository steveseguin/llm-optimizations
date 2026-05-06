# 2026-05-06 Q4_0 graph-pattern probe

## Context

The Q4_0 four-card communication and split-ratio sweeps are exhausted. The best quality-preserving path remains the three-card Q4_0 run at `46.194319 tok/s`, while four-card assist split reaches only `39.204149 tok/s`.

The next plausible software win is fewer launches and fewer repeated activation/quantization paths. llm-scaler points in the same direction: fused multi-GEMV and fused residual/norm/GEMV kernels for BMG.

## Patch

Patch artifact:

`/home/steve/llm-optimization-artifacts/patches/llama-cpp-meta-graph-pattern-stats-current-20260506.patch`

New env:

`GGML_META_GRAPH_PATTERN_STATS`

Modes:

- unset or `0`: no behavior change;
- `1`: print one summary per graph UID;
- `2`: print summary plus shared-activation MUL_MAT examples;
- `3`: print on every compute call, useful only for debugging graph rebuild churn.

## Validation Run

Log:

`/home/steve/bench-results/qwen36-q4_0-gguf/meta-graph-pattern2-triple213-p512n1-r1-20260506T175517Z.log`

JSONL:

`/home/steve/bench-results/qwen36-q4_0-gguf/meta-graph-pattern2-triple213-p512n1-r1-20260506T175517Z.jsonl`

Command shape:

- model: `Qwen3.6-27B-Q4_0.gguf`;
- devices: `SYCL0/SYCL1/SYCL2`;
- selector: `level_zero:2,1,3`;
- tensor split: `1/1/1`;
- prompt/output: `-p 512 -n 1`;
- flash attention: on;
- KV: f16;
- Q8 cache, fused MMVQ2, fused allreduce+ADD, and `GGML_SYCL_COMM_SYNC_AFTER=2` enabled.

## Results

Per graph UID:

- nodes: `3656`;
- `MUL_MAT`: `497`;
- partial `MUL_MAT`: `127-128`;
- Q4_0 `MUL_MAT`: `344`;
- `RMS_NORM -> MUL`: `209`;
- norm-fed matmul groups: `129`;
- norm-fed matmul edges: `369`;
- `ADD -> RMS_NORM -> MUL -> MUL_MAT*` groups: `128`;
- repeated-activation groups: `128`;
- matmuls in repeated-activation groups: `368`.

Representative examples:

- `attn_post_norm-N` feeds both `ffn_gate.weight:q4_0` and `ffn_up.weight:q4_0` in every layer.
- `attn_norm-N` feeds multiple attention-side projections. Some layers use combined `attn_qkv.weight:q4_0`; others expose separate `attn_q.weight:q4_0`, `attn_k.weight:q4_0`, and `attn_v.weight:q4_0`.
- Some `attn_norm-N` groups also include f32 SSM alpha/beta projections and a Q4_0 attention gate.

## Interpretation

The clean first target is FFN gate/up fusion for same-activation Q4_0 matmuls. It is present in every layer, shares `attn_post_norm`, and matches the llm-scaler fused multi-GEMV idea without needing to solve all attention projection layouts first.

Attention projection fusion is promising but more complex because Qwen3.6 alternates between combined and split Q/K/V layouts and mixes in non-Q4 f32 projections.

This is diagnostic only and was not submitted to LocalMaxxing.
