# 2026-05-08 MiniMax Branch-Fused Graph And Layer Timing

## Result

Tested an env-gated MiniMax graph construction change that builds attention and MoE FFN together per device branch:

```text
GGML_MINIMAX_GRAPH_BRANCH_FFN=1
```

This cut graph scheduler submissions from `503` splits/token to `251` splits/token.

Measured results on `4x Intel Arc Pro B70 32GB`, `unsloth/MiniMax-M2.7-GGUF` `UD-IQ4_XS`, RPC client plus four SYCL Level Zero workers:

| Mode | Tokens | tok/s | Notes |
| --- | ---: | ---: | --- |
| graph wave2 baseline | p0/n16 | 3.920533 | `503` splits/token |
| branch-fused graph, async off | p0/n1 | 2.309728 | `251` splits/token, `-sas 0` |
| branch-fused graph, wave async | p0/n1 | 4.593198 | `251` splits/token, `-sas 1` |
| branch-fused graph, wave async | p0/n16 | 5.634358 | `-sas 1` |
| branch-fused graph, wave async | p0/n64 | 5.639884 | `-sas 1` |
| branch-fused graph, async without wave | p0/n64 | 5.397710 | `-sas 1` |
| corrected layer baseline | p0/n64 | 14.292387 | still best valid MiniMax path |

Conclusion: branch fusion is a useful graph-path diagnostic improvement, but it is not competitive with layer mode yet. Keep it experimental and do not submit to LocalMaxxing.

## Patch

Patch snapshot:

```text
patches/ik-llama-minimax-rpc-device-map-and-graphsplit-20260508.patch
```

New behavior is isolated behind:

```text
GGML_MINIMAX_GRAPH_BRANCH_FFN=1
```

The default graph path is unchanged unless that env var is set.

## Repro

```bash
GGML_MINIMAX_GRAPH_BRANCH_FFN=1 \
GGML_SCHED_ASYNC_WAVE=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 1 -ngl 99 -sm graph -ts 1/1/1/1 \
  -fa 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 1 -o json
```

## Layer Timing Probe

Ran one layer-mode p0/n1 with `GGML_SYCL_OP_TIMING=1` on each RPC worker. This synchronizes every SYCL op, so the resulting `3.834472 tok/s` is not a performance score. It is a bottleneck attribution run.

Per-worker synchronized SYCL totals:

```text
rpc-b70-0 total_ms=61.355
rpc-b70-1 total_ms=62.520
rpc-b70-2 total_ms=60.582
rpc-b70-3 total_ms=64.212
```

Top aggregate op buckets across the four workers:

```text
116.615 ms   435  MUL_MAT
 29.484 ms    62  MOE_FUSED_UP_GATE
 25.804 ms   124  ROPE
 24.384 ms   124  CPY
 22.306 ms    62  SOFT_MAX
 11.240 ms   249  MUL
  7.581 ms    62  MUL_MAT_ID
```

Interpretation:

- Layer mode is reasonably balanced across the four B70s.
- The largest remaining MiniMax layer-mode targets are attention decode matmuls, RoPE, KV cache copies, softmax, and MoE fused up/gate.
- The first timing token includes cold kernels, so use the op mix and per-worker balance more than individual one-off layer spikes.

## Next

1. Keep branch-fused graph as an experiment, not a recommended runtime.
2. Investigate layer-mode attention decode kernels first: `kqv` matmul, RoPE, KV-cache copy, and softmax.
3. Consider a split/graph path only if it can avoid both the high RPC submission count and the current correctness/quality uncertainty around deferred reductions.
4. Retry LocalMaxxing for the corrected layer baseline when the API is reachable; do not submit the graph experiments.
