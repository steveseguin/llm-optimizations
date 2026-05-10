# MiniMax AOT Collective Inspection, 2026-05-10

## Why This Matters

The current MiniMax AutoRound path is no longer dominated only by the raw u4 MoE bridge. After the llm-scaler decode kernel work, the remaining gain has to come from graph-safe fusion around communication boundaries, Q/K RMS, attention/KV, projection, and MoE epilogues.

The latest public/docs check aligns with the local source:

- The `Lasimeri/MiniMax-M2.7-int4-AutoRound` model is a W4A16 AutoRound INT4 model with group size 128, which matches the current INC/WNA16 path.
- vLLM's current AllReduce+RMSNorm fusion target is a CUDA/FlashInfer path for Hopper/Blackwell and is advertised as a 5-20% low-token throughput improvement. That makes the idea relevant, but not directly reusable for Intel XPU/B70.
- Local vLLM XPU platform code forcibly disables `fuse_allreduce_rms`, `fuse_gemm_comms`, `enable_sp`, and related fusion passes on XPU.

## Local Graph Findings

Inspected AOT cache:

`/home/steve/.cache/vllm/torch_compile_cache/torch_aot_compile/4799a3c8468de261861723fba07480ef61e010f504245a62e5e93f4e9aef8e22/inductor_cache`

Counts from `scripts/summarize-vllm-aot-collectives.sh`:

- `all_reduce_comment_lines=40`
- `all_reduce_call_lines=36`
- `wait_tensor_call_lines=56`
- `rms_int4_lines=16`
- `fused_add_rms_lines=0`

Representative decoder-layer shape from `gc/cgcvvvkx...py`:

1. `int4_gemm_w4a16` computes a hidden projection into `f16[s72, 3072]`.
2. `_c10d_functional.all_reduce_` reduces that projection.
3. `_c10d_functional.wait_tensor` immediately fences the reduced projection.
4. A Triton kernel fuses residual add, RMSNorm math, and MoE/router preparation.
5. `vllm.moe_forward` runs the routed MoE.
6. The MoE output is allreduced and immediately waited.
7. A later Triton kernel fuses residual add, RMSNorm, and QKV `int4_gemm_w4a16`.
8. Q/K variance is computed into `f32[s72, 2]`, allreduced, and immediately waited before Q/K RMS application.

This explains why oneCCL algorithm and worker-affinity knobs have not helped much. The tiny allreduces are not expensive in isolation; the graph repeatedly forces allreduce plus wait boundaries directly in front of useful elementwise/RMS work.

## Patch Direction

Do not try to simply enable stock vLLM `fuse_allreduce_rms` on XPU. It will either be disabled by platform policy or fail because the implementation expects FlashInfer/AITER-style backends.

The first viable XPU patch should add a default-off B70/XPU-specific fused boundary:

1. Pattern-match `_c10d_functional.all_reduce_` + `wait_tensor` + residual add/RMSNorm in the compiled graph.
2. Replace it with an XPU custom op that performs XCCL/Level Zero compatible allreduce and the following residual/RMSNorm work with fewer visible graph boundaries.
3. Start with hidden-state post-attention or post-MoE boundaries before attempting Q/K RMS variance fusion.
4. Keep correctness guardrails strict: no skipped Q/K variance allreduce, no expert dropping, no root-residual shortcuts, and no speculative result unless target-verified.

## Reproduction

Run:

```bash
/home/steve/llm-optimizations-publish/scripts/summarize-vllm-aot-collectives.sh \
  /home/steve/.cache/vllm/torch_compile_cache/torch_aot_compile/4799a3c8468de261861723fba07480ef61e010f504245a62e5e93f4e9aef8e22/inductor_cache
```

Primary references checked:

- `https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound`
- `https://docs.vllm.ai/en/latest/features/torch_compile.html`
- Local source: `/home/steve/src/vllm/vllm/platforms/xpu.py`
- Local source: `/home/steve/src/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py`
