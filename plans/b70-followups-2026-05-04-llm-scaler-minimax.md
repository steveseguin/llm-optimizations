# B70 Follow-Ups: llm-scaler and MiniMax M2.7

Date: 2026-05-04

## Context

The current quality-preserving Qwen3.6 27B Q4_0 GGUF work has reached:

- 1x B70: 24.7 tok/s class
- 2x B70: 40.5 tok/s class with the Q8_1 activation-cache prototype
- 3x B70: 42.4 tok/s class with the Q8_1 activation-cache prototype
- 4x B70: still regresses to about 32 tok/s because tensor-parallel decode pays many small cross-device reductions

The four-GPU issue is currently treated as a software synchronization/communication problem, not a power-limit problem.

## Added Track: Intel llm-scaler Review

Local source path:

```text
/home/steve/src/llm-scaler
```

Repository state reviewed:

```text
origin/main e0b0703 2026-04-29 Update moe int4 prefill and decode (#384)
```

Initial findings to mine for B70 work:

- XPU `reduce_scatter`, `reduce_scatterv`, and `all_gatherv` support in the vLLM communicator patch.
- All-gather / reduce-scatter all-to-all path for expert-parallel communication.
- `SKIP_ALL_REDUCE` diagnostic in `RowParallelLinear`, useful for isolating row-parallel reduction cost.
- Custom ESIMD fused norm+GEMV INT4/FP8 kernels.
- QKV split + norm + RoPE fused kernel direction.
- Gated DeltaNet / conv decode kernels, relevant to Qwen3.6 recurrent layers.
- EAGLE/MTP-style GDN kernels, relevant to speculative decode exploration.
- oneDNN FP8 primitive caching and shape guards.
- GGUF batch dequantization concept, useful as a reference even though llama.cpp currently uses direct Q4_0 matvec kernels rather than full dequantization.

Working conclusion: `llm-scaler` is not yet assumed to be the runtime for Arc Pro B70, but it validates the next engineering direction: reduce, defer, or fuse the many small tensor-parallel reductions and move more decode-stage work into fused XPU kernels.

## 4-GPU Allreduce Trace

A one-token 4-GPU trace was run with the current SYCL single-kernel allreduce and Q8_1 activation cache enabled:

```text
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-allreduce-order-quad0123-q8cache-p0n1-20260504T164916Z.log
/home/steve/bench-results/qwen36-q4_0-gguf/sycl-allreduce-order-quad0123-q8cache-p0n1-20260504T164916Z.jsonl
```

Each measured pass reports:

- 128 allreduces
- every allreduce is 20,480 bytes
- ordering is two reductions per layer: `linear_attn_out-N` or `attn_output-N`, followed by `ffn_out-N`, for layers 0 through 63

Conclusion: simple adjacent packing is not the right first patch because these reductions are dependency-ordered through each layer. The next practical prototype should focus on delayed/fused reduction through safe graph regions, or fused row-parallel output kernels that consume partials and emit a mirrored result at the first true synchronization point.

## Added Track: MiniMax M2.7 Four-GPU Capacity Test

Model path provided by Steve:

```text
/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS
```

Local files are four GGUF shards totaling about 101G. This should barely fit in 128GB aggregate VRAM if tensor split overhead and KV cache are controlled.

First test should be conservative:

- use all four B70s with tensor split;
- pass the first GGUF shard to llama.cpp unless shard loading requires a different path;
- use tiny context and one generated token;
- use conservative KV/cache settings;
- record load stability, memory headroom, and shard handling before any performance claim;
- do not submit to LocalMaxxing until the load path and output sanity are stable.

## Updated Local Docs

Updated local files:

```text
/home/steve/b70-llm-lab-notes.md
/home/steve/q4_0-gguf-b70-optimization-plan.md
```
