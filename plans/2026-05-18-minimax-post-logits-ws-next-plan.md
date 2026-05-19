# MiniMax Post Logits-WS Next Plan

Date: 2026-05-18

## Current Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Current strict result: `87.279129` output tok/s, `116.372172` total tok/s
- Current recipe: clone-safe compiled custom allreduce on top of exact MiniMax logits-to-work-sharing llm-scaler INT4 MoE, no delayed attention allreduce, FlashAttention/PIECEWISE graph, MBT512.
- LocalMaxxing id: `cmpbsqm4l001qpc0199azisgz`

## Rules

- No quality-sacrificing shortcuts: every promoted path must pass raw token hashes, semantic canaries, repeated arithmetic, and extended canaries.
- No power-limit changes.
- Do not rerun already-negative branches unchanged: CCL fabric override, Q/K C10D group3, SP c10d pattern, IPC Q/K, distributed residual allreduce, or MoE-delay.
- Submit to LocalMaxxing only for valid, useful, quality-passed results.

## Next Branches

1. Completed: test `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` on top of the new logits-to-work-sharing baseline.
   - Reason: the older non-logits-WS combination was a near tie, but the exact combination with the new MoE path has not been quality-gated.
   - Outcome: strict quality passed, but mean speed was `81.021124` output tok/s and `108.028166` total tok/s, below the promoted `81.758267` baseline.
   - Decision: do not promote and do not submit to LocalMaxxing.

2. Completed: add better decode timing labels around residual allreduces and final logits boundaries.
   - Reason: the current timing helper labels Q/K variance allreduces and postprocess, but residual allreduces are not labeled precisely enough.
   - Outcome: diagnostic labels identified three similar steady decode collectives in eager mode: Q/K variance allreduce, attention delayed residual allreduce, and MoE expert output allreduce. Compiled post-forward timing showed final logits at about `0.86 ms/token`, mostly lm-head projection rather than TP gather.
   - Safety finding: model-forward timing wrappers were not neutral in compiled graph and were reverted from the active runtime. Active file hashes now match the promoted baseline again.

3. Use the timing results to choose a narrow fusion branch:
   - full-logits lm-head/postprocess boundary if exact token selection can be preserved;
   - hidden-state allreduce plus residual/add/norm boundary only if strict quality gates pass;
   - MoE/projection epilogue boundary, because MoE remains the largest eager per-layer region.

4. Completed: candidate-router repair branch.
   - Reason: test whether a smaller exact-router repair set can preserve routing quality while reducing overhead versus the exact MiniMax router-logits WS path.
   - Outcome: top16 failed the first raw145 n64 exact token-hash gate. Top32 passed the full strict gate but benchmarked at `80.008471` output tok/s and `106.677962` total tok/s, slower than the promoted `81.758267` / `109.011023` logits-WS baseline.
   - Decision: do not promote and do not submit to LocalMaxxing. Do not spend more time on smaller candidate repair sets unless a new implementation removes overhead without changing exact token selection.

5. Next active branch: MoE/projection epilogue scheduling.
   - Reason: repeated flag reties and router shortcuts have not beaten the exact logits-WS baseline; the decode timing points back to per-layer MoE/output collectives and epilogue work.
   - Method: inspect the llm-scaler INT4 MoE work-sharing kernel and Python wrapper, look for avoidable allocations/synchronization, output scaling/accumulation opportunities, and host-side dispatch overhead. Preserve exact top-k routing and expert weights.
   - Gate: smoke-test extension import and wrapper path, then run raw145 n64 before any benchmark. Promote only after the full strict gate and at least two benchmark repeats.
   - Update: static thread-local reuse of internal top-k and intermediate buffers was tested behind `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL=1`. It failed raw145 n64 with NUL/control-token corruption. Do not pursue static scratch reuse under XPU graph capture/replay without graph-safe lifetime handling.

6. Characterize prefill separately after decode remains stable.
   - Reason: LocalMaxxing followers asked for prefill numbers, but prefill tuning must not regress decode or quality.

7. Next diagnostic: MoE kernel trace on the promoted path.
   - Reason: the scratch-reuse failure shows allocation shortcuts can corrupt graph replay. Before changing more code, collect per-kernel wait timings with `LLM_SCALER_MOE_TRACE_KERNELS=1` on a short current-best run to decide whether top-k, up, down, or final logits is the next best exact target.
   - Constraint: trace mode is diagnostic only because it inserts waits and changes throughput.
   - Update: the first trace attempt was invalid because vLLM still compiled graph ranges; an enforced-eager trace completed and passed a short canary. Clean entries showed the expected WS up/down kernels, but multi-worker stderr interleaving makes aggregates approximate.
   - Follow-up tile sweep:
     - `VLLM_XPU_MOE_WS_UP_NTILE=4` passed full strict quality but slowed to `79.236469` output tok/s and `105.648625` total tok/s.
     - `VLLM_XPU_MOE_WS_UP_NTILE=8` stalled after graph compile.
     - `VLLM_XPU_MOE_WS_DOWN_HTILE=8` changed the exact raw token hash and was rejected without benchmark.
   - Decision: stop simple tile reties. Next work should target graph-safe MoE epilogue/work-sharing structure, final logits/lm-head cost, or cleaner non-invasive timing.

8. Completed: guarded greedy sampler fp32-skip.
   - Reason: the final logits timing showed about `0.86 ms/token`; a narrow sampler guard could skip `logits.to(torch.float32)` when greedy sampling has no logprobs, penalties, masks, or processors.
   - Outcome: full strict quality passed, but mean speed was `81.549421` output tok/s and `108.732562` total tok/s, slightly below the promoted `81.758267` / `109.011023` baseline.
   - Decision: reject and restore the active sampler to the promoted behavior. The sampler fp32 conversion is not the useful next bottleneck.

9. Next active branch: final lm-head/final-token selection or collective boundary work.
   - Reason: micro-reties around simple flags, tiles, and sampler conversion have not beaten the logits-WS baseline. The measured cost still points to local lm-head projection plus repeated per-layer collectives.
   - Method: avoid token-selection shortcuts unless they are exact under the strict gate. Prefer diagnostics or narrow patches that reduce kernel launches/copies around final logits, MoE output, or residual allreduce without changing routing, quantization, or sampling semantics.
   - Gate: raw145 n64 and n256 exact first, then semantic, arithmetic-repeat, extended sixpack, and at least two p512/n1536 benchmark repeats before promotion.
   - Update: retesting `MAX_BATCHED_TOKENS=768` on top of the newer clone-safe custom-allreduce recipe failed the extended sixpack with nondeterministic greedy output after passing the earlier gates. Do not pursue MBT reties unchanged; go back to exact collective-boundary/epilogue work.
   - Update: the functional out-of-place allreduce path (`VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1`) passed the full strict quality suite, but only reached `82.288077` output tok/s and `109.717436` total tok/s, below the promoted clone-safe custom allreduce result. Keep it as a quality-safe diagnostic baseline, not a promoted path.

10. Completed: MiniMax WS top-k-only reuse.
   - Reason: test a narrower graph-lifetime hypothesis than the earlier full internal scratch reuse failure by reusing only top-k tensors inside the MiniMax WS op.
   - Outcome: raw145 n64 exact failed immediately with NUL/control-token corruption. Reverting the patch and rebuilding restored the default raw145 n64 expected hash.
   - Decision: do not use static thread-local top-k reuse under XPU graph replay. This reinforces that allocation work needs graph-owned buffers rather than process-static tensors.
