# MiniMax No Timing Wrappers Retest

## Summary

Removed the local `timed_region(...)` wrappers from the active vLLM runtime
files used by MiniMax decode: tensor-parallel allreduce, attention, MoE runner,
MoeWNA16, and the GPU model runner. The timing helper module remains in the
tree for archived diagnostics, but the active inference path no longer imports
or calls it.

This is a cleanup and AOT-shape check, not a speed record. No weights, sampler
settings, speculative decoding, or GPU power limits were changed.

## Runs

| Label | Shape | Cache / AOT | KV tokens | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | --- |
| `20260510T102156Z` | p512/n512 cold | `4997925c5f` / `4799a3c8` | 9,408 | 28.05 | cold compile artifact |
| `20260510T102502Z` | p512/n512 warm | `4a91a7e50a` / `4799a3c8` | 17,216 | 35.85 | current floor |
| `20260510T102731Z` | p512/n1536 warm | `4a91a7e50a` / `4799a3c8` | 17,216 | 37.37 | current floor |

## Interpretation

Removing the timing wrappers did not recover the lost `41.130667 tok/s` AOT
schedule. It did keep the active runtime cleaner and the long-output warm run
was slightly better than the preceding clean-source p512/n1536 run, but still
within the current floor band.

The active runtime should keep these wrappers removed unless we deliberately
run a synchronized timing diagnostic. Future speed work should focus on actual
source-level work in Q/K collective fusion, attention/KV scheduling, or another
path that changes the per-token execution graph rather than more no-op wrapper
cleanup.

## Artifacts

- Logs: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-no-timing/`
- Cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-no-timing-wrappers-20260510T102156Z`
- Patch snapshot: `patches/vllm-active-minimax-no-timing-current-20260510.patch`
