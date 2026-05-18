# MiniMax Post Logits-WS Next Plan

Date: 2026-05-18

## Current Baseline

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Current strict result: `81.758267` output tok/s, `109.011023` total tok/s
- Confirmation repeat: `81.197954` output tok/s, `108.263938` total tok/s
- LocalMaxxing id: `cmpay7th600bbmn01v6csyaro`

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

2. Add better decode timing labels around residual allreduces and final logits boundaries.
   - Reason: the current timing helper labels Q/K variance allreduces and postprocess, but residual allreduces are not labeled precisely enough.

3. Use the timing results to choose a narrow fusion branch:
   - hidden-state allreduce plus residual/add/norm boundary,
   - MoE/projection epilogue boundary,
   - or final logits/postprocess boundary.

4. Characterize prefill separately after decode remains stable.
   - Reason: LocalMaxxing followers asked for prefill numbers, but prefill tuning must not regress decode or quality.
