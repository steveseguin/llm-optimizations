# Current B70 LLM Optimization Plan Update

Date: 2026-05-05

This note supersedes the stale portions of `plans/q4_0-gguf-b70-optimization-plan.md` in the artifact repo.

## Current Best Results

- Qwen3.6 27B Q4_0 GGUF, llama.cpp/SYCL, 3x B70 selector `2,1,3`: reshape-through-ADD fusion validation `44.812806 tok/s` decode, quality-preserving, software-only.
- Qwen3.6 27B static FP8, `vrfai/Qwen3.6-27B-FP8`, patched vLLM/XPU TP4 + FlashAttention2 + n-gram speculative decode: `47.674832 tok/s` decode, `95.349664 tok/s` total.
- Current FP8 best LocalMaxxing id: `cmos3pnqo0004kz04o4aiup22`.

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
- Runtime recovery after reboot:
  - `/etc/modprobe.d/xe-b70-headless.conf` sets `options xe disable_display=1 probe_display=0`;
  - all four B70s enumerate through Level Zero after reboot;
  - `/home/steve/sycl-peer-read-test` passes across all four GPUs;
  - upstream BMG GuC `70.49.4` is loaded on all four B70s.
- Q4_0 post-reboot validation:
  - exact fast 3x command shape survived prompt and decode;
  - 512 prompt / 512 output, 3 reps: prompt `135.705541 tok/s`, decode `44.180797 tok/s`, total `66.659637 tok/s`;
  - LocalMaxxing id: `cmoslhw0i0008jj04h59bb96n`;
  - note: `notes/2026-05-05-post-reboot-guc70494-q4-validation.md`;
  - data: `data/qwen36-q4-post-guc70494-validation-20260505.json`.
- Q4_0 kernel `6.17.0-23` revalidation:
  - same exact fast 3x command shape;
  - 512 prompt / 512 output, 3 reps: prompt `135.771640 tok/s`, decode `44.238455 tok/s`, total `66.735480 tok/s`;
  - output samples: `44.1095`, `44.3139`, `44.292 tok/s`;
  - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-guc70494-nondnn-triple213-exactfast-p512n512-r3-20260505T123032Z.jsonl`;
  - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-guc70494-nondnn-triple213-exactfast-p512n512-r3-20260505T123032Z.log`;
  - LocalMaxxing id: `cmosm05ke0005ib048aljq6pl`.
- Q4_0 reshape-through-ADD fusion:
  - graph trace fused all 48 `linear_attn_out -> RESHAPE -> ADD` decode sites into the existing allreduce+residual-add helper;
  - remaining decode allreduce paths: `127` `backend+add`, `1` plain `backend`;
  - 512 prompt / 512 output, 3 reps: prompt `135.806175 tok/s`, decode `44.812806 tok/s`, total `67.388784 tok/s`;
  - output samples: `44.8839`, `44.8199`, `44.7346 tok/s`;
  - JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-triple213-p512n512-r3-20260505T125442Z.jsonl`;
  - log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-reshapeadd-triple213-p512n512-r3-20260505T125442Z.log`;
  - LocalMaxxing id: `cmosmudwl0004k004hzz6l4u6`;
  - patch: `patches/llama-cpp-sycl-current-q4-reshapeadd-20260505.patch`.
- Post-reshape 4x/regression diagnostics:
  - 4x reshape-through-ADD trace fused the expected `48` reshape-through-ADD sites;
  - 4x allreduce summary worsened to `90.900 us` average for each 20KB f32 reduction, versus `47.632 us` on the 3x best path;
  - 4x root/order/helper screen degraded to roughly `23 tok/s` across root `0/1/2/3`, root-residual, skip-root-ready, and local-write variants;
  - current 3x subsets also degraded to roughly `32 tok/s` at `p512/n128`;
  - longer 3x `p512/n512/r1` recovered only to `40.699249 tok/s`, still below the validated `44.812806 tok/s`;
  - allreduce timing, Q4_0 reordered MMVQ dispatch, and GT0 boost clocks looked normal during the degraded state;
  - corrected PCIe AER `RxErr` / `BadTLP` events appeared on upstream bridge `0000:a1:00.0`;
  - disabling runtime PM / ASPM for a test did not recover performance;
  - PCI function reset of the B70 endpoints is unsafe on this stack: xe logged GT1 PF self-configuration timeouts and `sycl-ls` hung in uninterruptible sleep.
- Post-reboot recovery validation:
  - Level Zero enumerates all four B70s and `/home/steve/sycl-peer-read-test` passes;
  - 3x Q4_0 reshape-through-ADD, `p512/n512/r3`: prompt `135.469357 tok/s`, decode `45.624065 tok/s`, computed total `68.259384 tok/s`;
  - LocalMaxxing: `cmot9sgsi000lib042rqd6c62`;
  - clean 4x Q4_0 reshape-through-ADD, `p512/n512/r1`: prompt `102.210613 tok/s`, decode `34.375523 tok/s`, computed total `51.448022 tok/s`;
  - LocalMaxxing diagnostic negative-scaling result: `cmota1fpx0001l404wepbjtb7`;
  - immediate 3x health after clean 4x run remained good at `45.043267 tok/s` for `p512/n128`.
- Q4_0 allreduce + GET_ROWS fusion:
  - implemented `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` with SYCL proc `ggml_backend_comm_allreduce_get_rows_tensor`;
  - trace captured the intended `attn_output-63 -> GET_ROWS` site and replaced the remaining plain decode path with `backend+getrows`;
  - full validation, gate on, `p512/n512/r3`: prompt `135.626208 tok/s`, decode `45.375471 tok/s`, computed total `68.000508 tok/s`;
  - same-build gate-off control, `p512/n512/r3`: prompt `135.671512 tok/s`, decode `45.340867 tok/s`, computed total `67.967329 tok/s`;
  - conclusion: neutral at best and below the current `45.624065 tok/s` high-water mark; keep off by default and do not submit to LocalMaxxing as an improvement;
  - note: `notes/2026-05-06-q4-getrows-fusion-neutral.md`;
  - data: `data/qwen36-q4-getrows-fusion-20260506.json`;
  - patch: `patches/llama-cpp-sycl-meta-getrows-fusion-current-20260506.patch.gz.b64`.

## Interpretation

- Q4_0 4x scaling is not fixed by root selection, root-copy, local-write, or residual-read avoidance. The next useful work must reduce the number of tiny reductions or fuse communication into a lower-level matmul/reduction epilogue.
- FP8 TP4 is now the fastest validated single-session Qwen3.6 27B mode on this host, but adjacent n-gram flags are exhausted enough for now. Further FP8 work should target backend/runtime behavior rather than speculative flag sweeps.
- MiniMax M2.7 is currently blocked earlier than MoE/expert placement. The immediate issue is the SYCL `q8_0 x vector` dense attention matvec path on block 0.
- The runtime recovered after reboot. Clean 4x no longer wedges the runtime, but it remains slower than 3x. Avoid FLR/PCI reset for B70 recovery.

## Next Work

1. Runtime recovery:
   - treat reboot plus peer-read as the recovery procedure if xe/Level Zero degrades;
   - after any suspected runtime issue, confirm `sycl-ls` enumerates all four B70s and `/home/steve/sycl-peer-read-test` passes;
   - rerun the known-good Q4_0 3x validation before treating new benchmark data as valid;
   - record whether corrected PCIe AER on `0000:a1:00.0` returns under 3x/4x load;
   - do not use PCI function reset as a recovery method on this driver stack.
2. Q4_0 llama.cpp/SYCL:
   - use the post-reboot reshape-through-ADD 3x `45.624065 tok/s` run as the current control;
   - treat `attn_output-63 -> GET_ROWS` as inspected: a safe fused helper is possible, but timing is neutral and it should stay off by default;
   - stop spending time on 4x root/order/local-write sweeps unless a new collective implementation changes the tradeoff;
   - tune the 20KB f32 allreduce fast path only after the remaining graph-level fusions are exhausted;
   - prototype fewer reductions or a fused row-parallel output kernel only where mathematically safe;
   - investigate whether the row-parallel output projection can produce the mirrored post-allreduce output directly, eliminating the separate small collective rather than moving it later;
   - keep local-write/root-residual env gates diagnostic-only.
3. FP8 vLLM/XPU:
   - keep the PP2 x TP2 `self.drafter` getattr patch;
   - quarantine PP2+n-gram until stale speculative placeholder cleanup is fixed;
   - test real draft-model speculative decode if a compatible smaller Qwen draft model fits;
   - review oneCCL/XCCL options that affect TP4 latency without forcing the slower sockets/topology path.
4. MiniMax:
   - do not rerun MiniMax before Qwen recovery validation is clean;
   - stop treating the next blocker as MoE until the first dense q8_0 attention matvec is isolated;
   - build a small `q8_0 x vector` SYCL repro using the observed `[3072,6144]` by `[3072,1]` shape;
   - add focused traces or an env-gated fallback for q8_0 attention projections to verify whether the graph can reach MoE.
5. llm-scaler:
   - continue mining Intel `llm-scaler` for ideas around reduce-scatter/all-gather, fused norm+GEMV, Gated DeltaNet kernels, MTP/EAGLE kernels, and oneDNN FP8 primitive caching;
   - treat it as a reference source first, not a production backend assumption for Arc/B70.

## Submission Policy

- Submit only validated improvements or broadly useful diagnostics to LocalMaxxing.
- Do not submit the FP8 n-gram negative sweep as leaderboard results.
- Continue uploading patches, notes, and data artifacts to `steveseguin/llm-optimizations`.
