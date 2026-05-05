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
- Q4_0 allreduce-to-reshape fusion was technically correct but not a useful speed win:
  - 4x fused-add control, 512/128: `33.497463 tok/s`;
  - 4x fused-add plus reshape fuse, 512/128: `33.743952 tok/s`;
  - 3x fused-add plus reshape fuse, 512/512, 3 reps: `43.734996 tok/s`, below the fused-add-only validation at `44.004344 tok/s`.
- MiniMax M2.7 UD-IQ4_XS does not yet reach token generation:
  - scheduler tracing reaches split 1 on `SYCL0`;
  - SYCL op tracing completes `RMS_NORM` and elementwise `MUL`;
  - first blocker is `blk.0.attn_q.weight` `q8_0` `[3072,6144]` x `attn_norm-0` f32 `[3072,1]`;
  - default reordered MMVQ hangs after `quantize_row_q8_1_sycl`;
  - forced DMMV segfaults after `to_fp16_sycl`.
- Runtime recovery blocker after device-lost:
  - PCI reset of all four B70 VGA functions did not cleanly recover Level Zero;
  - `sycl-ls` then aborted in NEO DRM initialization at `drm_neo.cpp:445`;
  - `xe` unbind/rebind deadlocked during `0000:83:00.0` bind;
  - kernel stack is in `xe_display_init_early` / connector probing;
  - `/etc/modprobe.d/xe-b70-headless.conf` now sets `options xe disable_display=1 probe_display=0` for next boot.

## Interpretation

- Q4_0 4x scaling is not fixed by root selection, root-copy, local-write, or residual-read avoidance. The next useful work must reduce the number of tiny reductions or fuse communication into a lower-level matmul/reduction epilogue.
- FP8 TP4 is now the fastest validated single-session Qwen3.6 27B mode on this host, but adjacent n-gram flags are exhausted enough for now. Further FP8 work should target backend/runtime behavior rather than speculative flag sweeps.
- MiniMax M2.7 is currently blocked earlier than MoE/expert placement. The immediate issue is the SYCL `q8_0 x vector` dense attention matvec path on block 0.
- Current system state is blocked until reboot. Only one B70 is visible to Level Zero, with one `xe` bind task in uninterruptible kernel sleep and two B70s left unbound.

## Next Work

1. Q4_0 llama.cpp/SYCL:
   - after reboot, validate `sycl-ls` and `/home/steve/sycl-peer-read-test` across all four GPUs before full model runs;
   - rerun the known-good 3-card Qwen Q4 `p16/n8` smoke before changing code;
   - inspect reduction sites that remain after fused allreduce+ADD;
   - tune the 20KB f32 allreduce fast path before adding broader graph rewrites;
   - test root rotation, root-ready event skipping, fixed event vectors, and single-task/barrier alternatives for the tiny collective;
   - prototype fewer reductions or a fused row-parallel output kernel only where mathematically safe;
   - keep local-write/root-residual env gates diagnostic-only.
2. FP8 vLLM/XPU:
   - keep the PP2 x TP2 `self.drafter` getattr patch;
   - quarantine PP2+n-gram until stale speculative placeholder cleanup is fixed;
   - test real draft-model speculative decode if a compatible smaller Qwen draft model fits;
   - review oneCCL/XCCL options that affect TP4 latency without forcing the slower sockets/topology path.
3. MiniMax:
   - do not rerun MiniMax before Qwen recovery validation is clean;
   - stop treating the next blocker as MoE until the first dense q8_0 attention matvec is isolated;
   - build a small `q8_0 x vector` SYCL repro using the observed `[3072,6144]` by `[3072,1]` shape;
   - add focused traces or an env-gated fallback for q8_0 attention projections to verify whether the graph can reach MoE.
4. llm-scaler:
   - continue mining Intel `llm-scaler` for ideas around reduce-scatter/all-gather, fused norm+GEMV, Gated DeltaNet kernels, MTP/EAGLE kernels, and oneDNN FP8 primitive caching;
   - treat it as a reference source first, not a production backend assumption for Arc/B70.

## Submission Policy

- Submit only validated improvements or broadly useful diagnostics to LocalMaxxing.
- Do not submit the FP8 n-gram negative sweep as leaderboard results.
- Continue uploading patches, notes, and data artifacts to `steveseguin/llm-optimizations`.
