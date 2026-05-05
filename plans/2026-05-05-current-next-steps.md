# Current B70 LLM Optimization Plan Update

Date: 2026-05-05

This note supersedes the stale portions of `plans/q4_0-gguf-b70-optimization-plan.md` in the artifact repo.

## Current Best Results

- Qwen3.6 27B Q4_0 GGUF, llama.cpp/SYCL, 3x B70 selector `2,1,3`: `44.004344 tok/s` decode, quality-preserving, software-only.
- Qwen3.6 27B static FP8, `vrfai/Qwen3.6-27B-FP8`, patched vLLM/XPU TP4 + FlashAttention2 + n-gram speculative decode: `47.674832 tok/s` decode, `95.349664 tok/s` total.
- Current FP8 best LocalMaxxing id: `cmos3pnqo000kkz04o4aiup22`.

## Recent Negative Results

- Q4_0 4x local-write collective variants did not solve the 4-card regression:
  - root fused-add baseline: `33.219955 tok/s`;
  - local-write all-sites: `30.681785 tok/s`;
  - local-write fused-only: `32.365769 tok/s`;
  - root-residual reuse: `33.074515 tok/s`.
- FP8 nearby n-gram sweep did not beat the validated best:
  - n-gram `3`, lookup `2/4`: `40.697016 tok/s`;
  - n-gram `4`, lookup `2/3`: `43.130893 tok/s`;
  - n-gram `5`, lookup `2/4`: `44.163969 tok/s`.

## Interpretation

- Q4_0 4x scaling is not fixed by root selection, root-copy, local-write, or residual-read avoidance. The next useful work must reduce the number of tiny reductions or fuse communication into a lower-level matmul/reduction epilogue.
- FP8 TP4 is now the fastest validated single-session Qwen3.6 27B mode on this host, but adjacent n-gram flags are exhausted enough for now. Further FP8 work should target backend/runtime behavior rather than speculative flag sweeps.
- MiniMax M2.7 is currently blocked by llama.cpp split support for `minimax-m2` and split expert tensor allocation, not by total installed VRAM alone.

## Next Work

1. Q4_0 llama.cpp/SYCL:
   - inspect reduction sites that remain after fused allreduce+ADD;
   - prototype fewer reductions or a fused row-parallel output kernel where mathematically safe;
   - keep local-write/root-residual env gates diagnostic-only.
2. FP8 vLLM/XPU:
   - inspect and fix the PP2 x TP2 speculative drafter path where `XPUModelRunner` lacks `drafter`;
   - test real draft-model speculative decode if a compatible smaller Qwen draft model fits;
   - review oneCCL/XCCL options that affect TP4 latency without forcing the slower sockets/topology path.
3. MiniMax:
   - avoid full first-token runs until split expert tensor allocation is patched or isolated;
   - start with a synthetic split-expert allocation/dequant harness for `ffn_down_exps` / `iq4_xs`.
4. llm-scaler:
   - continue mining Intel `llm-scaler` for ideas around reduce-scatter/all-gather, fused norm+GEMV, Gated DeltaNet kernels, MTP/EAGLE kernels, and oneDNN FP8 primitive caching;
   - treat it as a reference source first, not a production backend assumption for Arc/B70.

## Submission Policy

- Submit only validated improvements or broadly useful diagnostics to LocalMaxxing.
- Do not submit the FP8 n-gram negative sweep as leaderboard results.
- Continue uploading patches, notes, and data artifacts to `steveseguin/llm-optimizations`.
