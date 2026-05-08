# 2026-05-08 MiniMax Correct Graph Reduce Diagnostic

## Summary

The earlier MiniMax graph experiments were useful for speed exploration, but the deferred-reduce path was not quality-safe. The key issue is that MiniMax layer boundaries contain nonlinear operations, so a deferred partial sum cannot be carried through RMSNorm/router/MoE and reduced later:

```text
RMSNorm(sum(shards)) != sum(RMSNorm(shards))
```

I added an opt-in diagnostic path to force real reductions at the attention and FFN boundaries:

```text
GGML_MINIMAX_NO_DEFER_REDUCE=1
GGML_RPC_REDUCE_MIRROR=1
```

This run succeeded, which confirms the corrected dataflow can execute through the current RPC + SYCL stack. It is not a speed result yet.

## Result

Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`

Hardware: `4x Intel Arc Pro B70 32GB`

Mode: RPC client + four SYCL/Level Zero RPC workers, graph split, f16 KV, `-nkvo 0`

| Run | Tokens | tok/s | Notes |
| --- | ---: | ---: | --- |
| quality-correct graph reduce smoke | p0/n1 | 2.033524 | real reduce at nonlinear boundaries, mirrored result to source shards |
| branch-fused graph, earlier | p0/n64 | 5.639884 | faster but not promoted because deferred reductions cross nonlinear regions |
| current valid layer baseline | p0/n64/r3 | 16.383602 | LocalMaxxing `cmowft2hr000oo3019is4snoq` |

The corrected graph path produced `745` scheduler splits for one token in the first trace attempt. That is the current bottleneck: the path is doing many small RPC submissions and real reductions through the client instead of a low-latency device-side collective.

## Patch

Patch snapshot:

```text
patches/ik-llama-minimax-rpc-device-map-and-graphsplit-20260508.patch
```

Relevant opt-in changes:

- `GGML_MINIMAX_NO_DEFER_REDUCE=1` disables the MiniMax graph path's deferred reduce marker at attention and FFN boundaries.
- `GGML_RPC_REDUCE_MIRROR=1` mirrors the client-side reduced tensor back to each source shard after a real reduce.
- `get_input_tensor_sm_graph()` now returns the materialized reduce tensor when a reduce is not deferred; without this fix, downstream nonlinear nodes could still consume per-device partials.

All of this is diagnostic and default-off.

## Repro

```bash
GGML_MINIMAX_NO_DEFER_REDUCE=1 \
GGML_RPC_REDUCE_MIRROR=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 1 -r 1 -ngl 99 -sm graph -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

Actual result file:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/rpc-graph-nodefer-mirror-inputfix-p0n1-20260508T063928Z.jsonl
```

The JSON stream is polluted by worker memory-print lines before the JSON object, so parse it defensively instead of piping directly to `jq`.

## Interpretation

This closes the uncertainty around graph-mode correctness: a quality-correct graph path must reduce before every nonlinear boundary and broadcast/mirror the full reduced activation before the next branch. The current implementation proves the dataflow, but it is too slow because reductions are host-mediated through the RPC client.

The next >30 tok/s MiniMax route should therefore avoid spending more time on naive graph split reduction. Better candidates are:

1. Keep layer mode valid and improve the largest local kernels: attention `MUL_MAT`, RoPE, KV copy, softmax, and `MOE_FUSED_UP_GATE`.
2. Build a lower-overhead active-expert combine path that reduces CPU/RPC orchestration without changing math.
3. Revisit graph/tensor parallelism only with a lower-latency allreduce/broadcast design; client-side real reduce is the wrong steady-state primitive.
