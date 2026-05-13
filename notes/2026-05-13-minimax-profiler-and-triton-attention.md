# MiniMax Profiler And Triton Attention Screen, 2026-05-13

## Purpose

Continue the MiniMax M2.7 AutoRound optimization work without lowering quality.
The guardrails stayed fixed: no model, quantization, router precision, KV dtype,
sampler, speculative decoding, or power changes.

The current promoted recipe remains TP4 FP16 on 4x Intel Arc Pro B70 with
llm-scaler INT4 MoE, XPU graph, MiniMax attention delayed allreduce,
`--block-size 256`, `MAX_BATCHED_TOKENS=512`, prefix caching disabled, and
p512/n1536 measurement.

## Current Best

The standing result is still the repeat high:

- output tok/s: `73.306312`
- total tok/s: `97.741749`
- AOT hash:
  `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`
- LocalMaxxing: `cmp4f31dh000amz01uqa329e6`

## Profiler Diagnostic

I ran a p512/n32 profiler pass on the current best recipe. The profiler adds
heavy overhead, so the tok/s from this run is not used as a benchmark. It is
only diagnostic.

Rank 0 highlights:

| Event | Calls | Time |
| --- | ---: | ---: |
| `_vllm_fa2_C::varlen_fwd` | `496` | `7.410 ms` self XPU, `89.188 ms` CPU total |
| `gemm_kernel` | `15` | `5.055 ms` self XPU |
| `fused_moe_kernel_gptq_awq` | `5` | `2.315 ms` self XPU |
| `oneccl_allreduce_pcie<half...>` | `5` | `644.060 us` self XPU |
| `oneccl_allgatherv_pcie<half...>` | `9` | `428.642 us` self XPU |
| `zeEventHostSynchronize` | `464` | `54.342 ms` self CPU |
| `urEnqueueKernelLaunch` | `664` | `14.942 ms` self CPU |
| `vllm::unified_attention_with_output` | `496` | `104.316 ms` CPU total |

Interpretation: the current FlashAttention path is still an important
PIECEWISE graph boundary. Attention itself is not the only cost; the host-side
sync and launch activity around the boundary is visible enough to justify more
source-level work.

## Triton FULL Graph Screen

I tested `--attention-backend TRITON_ATTN` with `cudagraph_mode=FULL`, keeping
the same model, quantization, dtype, block size, prefix-cache-off setting, and
p512/n1536 measurement shape.

The Triton run did capture a FULL graph cleanly:

- AOT hash:
  `20ce3b498033eac955acbe2a8e473a337b2159fa71664dde2b53c12ea60724b2`
- log confirmed `Using Triton backend.`
- log confirmed `cudagraph_mode: FULL`
- graph capture finished in 2 seconds and took `0.02 GiB`

Results:

| Run | Output tok/s | Total tok/s | Decision |
| --- | ---: | ---: | --- |
| Current best p512/n128 control | `46.591756` | `232.958781` | comparison baseline |
| Triton FULL p512/n128 | `55.969679` | `279.848396` | short-output screen only |
| Triton FULL p512/n1536 | `71.689965` | `95.586621` | negative |

The p512/n128 result shows Triton FULL is genuinely faster at short output
lengths: `+9.377923` output tok/s over the current-best control. The p512/n1536
result is `1.616346` output tok/s slower than the current best, so Triton FULL
graph is not promoted for sustained decode.

## AOT Collective Check

The Triton AOT graph has the same collective shape as the current-best
FlashAttention graph:

- actual allreduce lines: `1496`
- waits: `1496`
- wait gap: all waits were exactly 2 generated-code lines after allreduce
- categories:
  - embedding hidden allreduce: `8`
  - Q/K RMS variance allreduce: `496`
  - attention output projection hidden allreduce: `496`
  - MoE hidden allreduce: `496`

This explains why Triton FULL graph did not unlock a large sustained-decode
gain. It improved graph capture behavior but did not reduce the collective
structure that dominates the TP4 MiniMax path.

## Decision

Keep the FlashAttention/PIECEWISE recipe as the current best. The next useful
work is not another broad scheduler flag sweep; it is source-level work around
the 187 collectives per generated decode graph:

1. hidden-state allreduce plus residual/RMSNorm or projection/MoE epilogue
   fusion,
2. Q/K RMS variance allreduce fusion only if target math is preserved,
3. attention boundary cleanup that reduces host synchronization without losing
   the faster FlashAttention kernel.

## Artifacts

- Data summary:
  `data/minimax-m27-profiler-and-triton-attention-20260513.json`
- Profiler log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n32-20260513T185810Z.log`
- Profiler directory:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/profiler-current-best-20260513T185810Z`
- Triton p512/n1536 log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T192211Z.log`
- Current-best p512/n128 control log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T193330Z.log`
