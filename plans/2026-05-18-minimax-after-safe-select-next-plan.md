# MiniMax After Safe-Select Next Plan

Date: 2026-05-18

## Position

The promoted MiniMax M2.7 AutoRound result remains `81.758267` output tok/s and `109.011023` total tok/s on 4x B70, with strict quality gates passed. The latest safe hidden-state selection candidate passed quality but slowed to `77.314354` output tok/s, so it is rejected.

## Constraints

- Preserve output quality: exact raw token hashes, semantic canaries, 16-repeat arithmetic, and extended sixpack before any promoted benchmark.
- Do not change GPU power limits.
- Do not rerun unchanged negative branches: local argmax variants, sampler fp32 skip, static scratch reuse, simple MoE tile reties, candidate-router top16/top32, MoE delay, no-clone reties, CCL fabric override, and Q/K C10D/SP/IPC experiments.
- Submit to LocalMaxxing only for useful quality-passed results; negative or neutral findings stay in notes/data.

## Next Work

1. Final lm-head/projection boundary:
   - Reason: synchronized timing still points to about `0.86 ms/token` in final logits, mostly local lm-head projection rather than logits gather.
   - Action: inspect whether the final-token lm-head path can avoid work that is not needed for greedy single-token decode while preserving exact token selection. Do not use approximate local argmax shortcuts.
   - Gate: raw145 n64/n256 exact first, then full strict gate if promising.

2. Graph-safe MoE epilogue and output collective:
   - Reason: allocation reuse under process-static/thread-local buffers corrupted XPU graph replay, but MoE output and allreduce remain decode-critical.
   - Action: inspect llm-scaler INT4 WS up/down kernels and Python wrapper for launch count, host synchronization, temporary allocation, and output scaling/accumulation boundaries. Prefer graph-owned or request-owned buffers if any allocation reduction is attempted.
   - Gate: extension import smoke, raw145 n64 exact, then full strict gate.

3. Non-invasive timing cleanup:
   - Reason: compiled model-forward wrappers were not neutral. Need lower-risk timing around final logits and per-layer collectives without changing captured graph semantics.
   - Action: use existing logs, kernel trace in enforced-eager only, or wrapper timing outside captured regions. Treat timing runs as diagnostic-only.

4. Prefill characterization:
   - Reason: followers asked for prefill numbers, and prefill may be optimized separately.
   - Action: collect p512/n1536 total/prefill/decode and a short p2048/n256 characterization without changing the promoted decode recipe.
   - Constraint: do not promote any prefill change that slows decode or fails quality.

5. Reproducibility hygiene:
   - Record each candidate in notes/data/patches.
   - Keep strict runner environment summaries complete.
   - Publish useful results through the GitHub connector and LocalMaxxing only when warranted.
