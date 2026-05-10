# MiniMax Q/K Pass XPU Helper

## Summary

Added an opt-in XPU fallback to vLLM's existing MiniMax Q/K norm fusion pass.
The upstream-style pass normally prefers CUDA `minimax_allreduce_rms_qk` and a
Lamport CUDA workspace. On this XPU-only B70 system that path fails with
`cudaErrorInsufficientDriver`, so the local patch adds
`VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1` to:

- skip the CUDA Lamport workspace;
- register a `vllm.minimax_qk_norm_fused_xpu_helper` custom op backed by the
  existing `minimax_qk_rms_xpu` helper extension;
- express the pass pattern with `torch.ops.vllm.all_reduce(..., group_name)` to
  avoid torchbind group objects during pattern generation.

This keeps MiniMax model source clean and makes the Q/K helper a compiler-pass
experiment. It is disabled by default.

## Runs

| Label | Shape | Cache / AOT | KV tokens | Output tok/s | Result |
| --- | --- | --- | ---: | ---: | --- |
| `20260510T103321Z` | p1/n8 | CUDA workspace not skipped | n/a | n/a | failed: `cudaErrorInsufficientDriver` |
| `20260510T103612Z` | p1/n8 | workspace skipped, old pattern | n/a | n/a | failed: torchbind pattern object |
| `20260510T104145Z` | p1/n8 | `cf4951af` | 9,408 | 2.73 | liveness pass |
| `20260510T104447Z` | p512/n512 cold | `e5d63204a4` / `2655f1c8` | 9,408 | 28.25 | cold compile artifact |
| `20260510T104756Z` | p512/n512 warm | `b624e23ba9` / `2655f1c8` | 17,216 | 36.44 | current floor |
| `20260510T105026Z` | p512/n1536 warm | `b624e23ba9` / `2655f1c8` | 17,216 | 37.24 | current floor |

## Interpretation

The pass now compiles and runs on XPU, which is useful for future fusion work,
but the helper-backed replacement does not recover the prior `41.130667 tok/s`
high. The likely reason is that this fallback still performs local variance
kernels, a normal oneCCL allreduce, and apply kernels; it changes the pass
plumbing but not the underlying per-token collective/kernel structure enough to
matter.

Keep `VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION` unset for real benchmarks. The
next Q/K path should either implement a real fused XPU op with communication
inside the kernel/extension, or target a different attention/KV scheduling
bottleneck.

## Artifacts

- Logs: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qk-pass/`
- Cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-qk-pass-xpu-helper-p512-20260510T104447Z`
- Patch: `patches/vllm-minimax-qk-pass-xpu-helper-20260510.patch`
