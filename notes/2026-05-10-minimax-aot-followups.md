# MiniMax AOT Follow-ups After c158 Regression

## Summary

These follow-ups were run after the favorable `c15860...` MiniMax AOT payload
was lost/regenerated. The accepted LocalMaxxing high remains the valid
`41.130667` output tok/s p512/n1536 run. The current reproducible path is
closer to `36-37` output tok/s.

No model weights, sampling settings, speculative decoding, or GPU power limits
were changed.

## Results

| Label | Shape | Change | Cache / AOT | Output tok/s | Result |
| --- | --- | --- | --- | ---: | --- |
| `20260510T065638Z` | p512/n512 | extra disabled `timed_region` boundaries around layer norm, attention, and MoE | `151eae70b1` / `d6e4bdfe` | 28.03 | negative |
| `20260510T070155Z` | p512/n512 | archived larger `c15860...slow-after-timing-noop` AOT copied into isolated cache | `1d97049441` / `c15860dd` | 35.22 | not old fast binary |
| `20260510T070438Z` | p512/n1536 | live default cache repeat | `1d97049441` / `c15860dd` | 37.05 | current floor repeat |
| `20260510T070856Z` | p512/n512 | `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` retest | `1d97049441` / `c15860dd` | 36.13 | no recovery |
| `20260510T071210Z` | p512/n512 | aggressive Inductor autotune, default memory | `8e4f15faa1` / `c15860dd` | n/a | failed: negative KV headroom |
| `20260510T071738Z` | p512/n512 | aggressive Inductor autotune cache reused with `gpu_memory_utilization=0.95` | `5c94c78739` / new compile | 23.15 | very negative |

## Interpretation

Adding more generic timing boundaries does not recreate the old favorable
compiled graph. The new three-subgraph family (`0,1,62`) is slower, and the
explicit layer-level boundaries still compiled into the slower shape while
reducing available KV memory to 9,408 tokens.

The archived larger `c15860...slow-after-timing-noop` directory is not the lost
fast binary. It starts and keeps the four-subgraph graph shape, but it only
reaches 35.22 output tok/s.

The oneCCL topology-recognition bypass no longer reproduces the earlier 39 tok/s
p512/n512 screen. It now lands at 36.13 output tok/s, so the lost speed was not
just that environment variable.

Aggressive Inductor autotune is not a recovery path on this stack. It produced a
large compiled payload with no KV headroom at default memory. Retrying from the
same cache with `gpu_memory_utilization=0.95` caused another graph compile and
fell to 23.15 output tok/s.

## Next Work

- Keep all compile-shape experiments in isolated `VLLM_CACHE_ROOT` directories.
- Keep the current live MiniMax source on the graph-shaped `c15860...` floor
  patch until a deterministic faster path is found.
- The most promising quality-preserving source path is still a true XPU
  equivalent of vLLM's CUDA `minimax_allreduce_rms_qk`, or another graph-safe
  way to fuse Q/K variance, TP exchange, and RMS apply without changing MiniMax
  math.
- Do not submit these follow-up runs to LocalMaxxing; they are diagnostic
  regressions or current-floor repeats below the accepted high.
