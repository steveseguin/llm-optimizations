# MiniMax M2.7: AOT Cache and Graph-Shape Regression

## Summary

After the post-reboot restore, the fast MiniMax AutoRound FP16 TP4 path became
sensitive to the exact vLLM compiled graph and AOT cache artifact. The prior
accepted high was `41.130667` output tok/s for p512/n1536, but later graph
inspection moved that AOT artifact to suspect status because the Q/K RMS
variance allreduce was not visible in the fast graph. The current
quality-conservative p512/n1536 reference is `37.552538` output tok/s.

No model weights, quantization, sampling, speculative decoding, or GPU power
settings were changed in these screens.

## Runs

| Label | Shape | Change | Cache / AOT | Output tok/s | Result |
| --- | --- | --- | --- | ---: | --- |
| `20260510T060419Z` | p512/n512 | timing helper no-op experiment | `b3633d6ffb` / `c15860dd` | 36.18 | negative |
| `20260510T061025Z` | p512/n512 | helper reverted, stale regenerated AOT | `1d97049441` / `c15860dd` | 35.75 | below old fast artifact |
| `20260510T061546Z` | p512/n512 | moved AOT, fresh rebuild | `b5636051eb` / `c15860dd` | 35.64 | no recovery |
| `20260510T062032Z` | p512/n512 | `VLLM_DISABLE_COMPILE_CACHE=1` | none persisted | 35.58 | no recovery |
| `20260510T062607Z` | p512/n512 | removed default-off MiniMax timing/QK/router branches | `dca8eb83bf` / `8ab1171d` | 28.64 | very negative |
| `20260510T063116Z` | p512/n512 | restored timing/QK branches but synced source K-norm branch | `6fadc05593` / `221552de` | 26.69 | very negative |
| `20260510T063512Z` | p512/n512 | restored simple K-norm graph shape and archived `c15860dd` AOT | `1d97049441` / `c15860dd` | 35.50 | floor recovered |
| `20260510T063845Z` | p512/n512 | isolated cache, max autotune and coordinate descent disabled | `00a89c9e85` / `6d8ea006` | 27.47 | negative |
| `20260510T064158Z` | p512/n512 | isolated cache, combo kernels disabled | `dec3ebf310` / `660e9600` | 28.32 | negative |
| `20260510T064500Z` | p512/n512 | current floor, `gpu_memory_utilization=0.95` | `1d97049441` / `c15860dd` | 35.64 | neutral |
| `20260510T064725Z` | p512/n1536 | current floor, `gpu_memory_utilization=0.95` | `1d97049441` / `c15860dd` | 36.56 | below accepted high |

## Interpretation

The previous `39-41` tok/s MiniMax path was not only a source-level state. It
depended on a favorable persisted AOT artifact under the `c15860...` hash. The
timing-helper no-op experiment regenerated that AOT family and the old fast
binary appears to have been overwritten.

The timing wrappers in `minimax_m2.py` are not pure diagnostics for compiled
throughput. Removing them changed vLLM/Inductor graph partitioning and produced
a much slower AOT family. Keep those wrappers in place until there is a better
explicit graph-partitioning strategy.

The K-norm constructor branch that is logically equivalent for this exact model
also changes the compiled graph hash. For MiniMax M2.7 with 8 KV heads and TP4,
the simple K-norm construction is functionally equivalent and recovers the
known `c15860...` AOT family. Revisit the more general replicated-KV branch only
when testing a model whose KV-head count is smaller than TP size.

Disabling Inductor max autotune, coordinate-descent tuning, or combo kernels is
not a speed path. Both isolated-cache screens were much slower and reduced KV
headroom to 9,408 tokens.

## Current State

The active runtime is back on the quality-preserving MiniMax TP4 path:

- vLLM/XPU, FP16 activations
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- llm-scaler u4 decode-only MoE bridge
- default oneCCL IPC/topology
- XPU graph disabled
- normal Q/K TP allreduce enabled
- simple K-norm constructor for this MiniMax M2.7 TP4 graph shape

Do not submit these regression/floor runs to LocalMaxxing as new records. The
accepted `41.130667` p512/n1536 run should be treated as a suspect scheduling
clue, and the later `37.552538` p512/n1536 Q/K-allreduce run is the current
quality-conservative public reference.

## Next Work

- preserve the current `c15860...` floor cache and use isolated `VLLM_CACHE_ROOT`
  for all compile-shape experiments;
- look for a deterministic way to reproduce the old favorable AOT schedule,
  possibly by inspecting generated Inductor artifacts and `.best_config` files;
- move from incidental graph shaping via `timed_region` to explicit
  source-level fusion or graph partitioning around Q/K RMS, RoPE, attention, and
  TP collectives.
