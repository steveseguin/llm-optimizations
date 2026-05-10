# MiniMax Active Clean Source Retest

## Summary

After the IPC and helper experiments, the active MiniMax vLLM model source was
reduced back to a minimal patch: keep only the K-norm tensor-parallel
replication fix in `minimax_m2.py`; remove default-off Q/K helper, Q/K
apply+RoPE helper, FP16-router, IPC, skip-allreduce, contiguous-copy, and timing
branches from the active runtime path.

This is a runtime hygiene result, not a speed record. It confirms the lost
`39-41 tok/s` schedule is not recovered merely by removing dormant diagnostic
branches from the model wrapper.

No model weights, sampling behavior, speculative decoding, or GPU power limits
were changed.

## Runs

| Label | Shape | Cache / AOT | KV tokens | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | --- |
| `20260510T100729Z` | p512/n512 cold | `2d59d0e621` / `4799a3c8` | 9,408 | 27.47 | cold compile artifact |
| `20260510T101040Z` | p512/n512 warm | `cf667763cb` / `4799a3c8` | 17,216 | 36.14 | current floor |
| `20260510T101309Z` | p512/n1536 warm | `cf667763cb` / `4799a3c8` | 17,216 | 36.63 | current floor |

## Interpretation

The clean source direct-loads from the new isolated AOT cache and restores full
KV headroom on warm runs. Throughput remains in the same `36-37 tok/s` band as
the previous current-floor runs. That rules out dormant helper/router branches
as the main cause of the missing `41.130667 tok/s` LocalMaxxing high.

The active runtime should stay on the clean K-norm-only MiniMax source while
future experiments live as archived patches or isolated branches. The next
quality-preserving speed work remains a real graph-safe Q/K collective fusion or
another source-level way to reduce attention/projection scheduling overhead.

## Artifacts

- Logs: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-clean-source/`
- Cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-clean-source-20260510T100729Z`
- Patch: `patches/vllm-minimax-active-clean-knorm-only-20260510.patch`
