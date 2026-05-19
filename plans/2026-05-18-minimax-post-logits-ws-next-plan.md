# MiniMax Post Logits-WS Next Plan

Date: 2026-05-18

## Current Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Current strict result: `88.748424` output tok/s, `118.331232` total tok/s
- Current recipe: clone-safe compiled custom allreduce on top of exact MiniMax logits-to-work-sharing llm-scaler INT4 MoE, no delayed attention allreduce, FlashAttention/PIECEWISE graph, MBT512, with `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2` to skip the pre-custom-op clone only for tiny FP32 Q/K variance allreduces.
- LocalMaxxing id: `cmpbz7lyc004rpc019jburzqv`. Previous clone-safe custom-allreduce baseline: `cmpbsqm4l001qpc0199azisgz`.

## Rules

- No quality-sacrificing shortcuts: every promoted path must pass raw token hashes, semantic canaries, repeated arithmetic, and extended canaries.
- No power-limit changes.
- Do not rerun already-negative branches unchanged: CCL fabric override, Q/K C10D group3, SP c10d pattern, IPC Q/K, distributed residual allreduce, MoE-delay, MBT reties above 512, or static scratch reuse under XPU graph replay.
- Submit to LocalMaxxing only for valid, useful, quality-passed results.

## Completed Findings

1. Clone-safe compiled allreduce custom-op was the large current win.
   - `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` plus `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1` passed full strict quality and reached `87.279129` output tok/s.
   - Direct custom-op without alias protection failed raw exact quality.

2. Shape-gated tiny-FP32 clone elision is now promoted.
   - `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2` passed raw145 n64/n256 exact hashes, semantic suite, 16-repeat arithmetic, and extended sixpack.
   - Four-run mean: `88.748424` output tok/s and `118.331232` total tok/s.
   - Caveat: PyTorch emits a custom-op alias warning on the tiny no-clone path. Treat this as technical debt, not a final upstreamable design.

3. Graph-visible clone custom-allreduce was rejected.
   - `VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1` with internal custom-op clone disabled completed AOT graph compilation but hung before producing the first raw145 n64 quality JSON.
   - Do not pursue this exact path unchanged.

4. Functional out-of-place allreduce was quality-safe but slow.
   - `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1` passed full strict quality but only reached `82.288077` output tok/s.
   - Keep it as a diagnostic baseline, not a promoted path.

5. MBT reties above 512 are not useful on the current path.
   - MBT768 passed early gates but failed the extended sixpack with nondeterministic greedy output on the sort/list prompt.
   - MBT832 was slower in an earlier strict run; MBT896/1024 produced quality corruption.

6. Static graph scratch reuse is unsafe.
   - Full internal reuse and top-k-only reuse produced NUL/control-token corruption under XPU graph replay.
   - Future allocation work needs graph-owned buffers or explicit graph lifetime handling.

7. Router/logits shortcuts have not beaten the exact path.
   - Candidate-router top32 was quality-safe but slower.
   - Local argmax and sampler fp32-skip were quality-safe in guarded cases but slower after the exact logits-to-WS MoE path became active.

8. Timing still points at collective boundaries and epilogues.
   - Eager labels identified three similar steady decode collectives: Q/K variance allreduce, attention delayed residual allreduce, and MoE expert output allreduce.
   - Compiled post-forward timing found final logits around `0.86 ms/token`, mostly lm-head projection rather than TP gather.

## Next Branches

1. Make the tiny-FP32 allreduce path alias-correct.
   - Reason: the new win proves clone overhead on the scalar FP32 Q/K variance collective matters, but the current implementation relies on behavior PyTorch warns will become invalid.
   - Method A: add or locate an out-of-place/non-aliasing custom allreduce return path for very small FP32 tensors so the graph sees a distinct output without paying the general Python-side clone cost.
   - Method B: specialize the vLLM custom allreduce wrapper to return a fresh output only for the tiny FP32 path.
   - Gate: raw145 n64/n256 exact first, then semantic suite, 16-repeat arithmetic, extended sixpack, and at least four p512/n1536 benchmark repeats before replacing the current result.

2. Fuse MiniMax Q/K variance allreduce with RMS apply.
   - Reason: Q/K RMS variance allreduce is now confirmed decode-sensitive.
   - Preserve exact restored weights and token hashes; no approximate RMS, no changed dtype policy, and no hidden router changes.
   - This is likely more useful than another wrapper-level clone retie.

3. Revisit MoE output allreduce plus residual/RMS epilogue fusion.
   - Reason: MoE remains the largest per-layer region, but simple tile reties and static scratch reuse failed.
   - Look for graph-safe epilogue fusion or a more exact integration point in llm-scaler/vLLM rather than changing expert selection or route weights.

4. Characterize prefill separately after decode remains stable.
   - Reason: LocalMaxxing followers asked for prefill numbers, but prefill tuning must not regress decode or quality.
   - Use separate reporting; do not let total tok/s hide a decode regression.

5. Keep speculative decoding optional.
   - It may help user-visible speed, but it is not a replacement for the exact decode-path optimization work and must pass quality checks separately.
