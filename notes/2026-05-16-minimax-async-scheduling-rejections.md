# MiniMax M2.7 Async Scheduling Rejections

Date: 2026-05-16

Hardware: 4x Intel Arc Pro B70 32GB

Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

## Context

The quality-valid MiniMax path is still the no-async TP4 XPU graph recipe:

- Mean decode: 62.662 tok/s
- Mean total: 83.550 tok/s
- LocalMaxxing id: `cmp7sf8xo00dko4018nmrmckc`
- Quality gate: raw145 n64/n256 exact hashes, semantic suite, and extended six-prompt repeat canary

The faster async-scheduling path previously reached roughly 66 tok/s but failed repeatability with token-id corruption. I tested narrower fixes to see whether we could recover that speed without sacrificing quality.

## Results

### XPU stream/event async output copy

Candidate: `xpu-async-copy-stream-fix`

Change tested:

- Use `torch.xpu.Stream`, `torch.xpu.Event`, `torch.xpu.current_stream`, and `torch.xpu.stream` for XPU async output copy instead of CUDA stream/event APIs.

Outcome:

- raw145 n64 exact: pass
- raw145 n256 exact: pass
- semantic repeat suite: fail
- Failure: nondeterministic token hashes
- Observed drift: arithmetic canary generated `42` in one repeat and ` 42` in another repeat.
- No benchmark run. No LocalMaxxing submission.

### XPU sampled-token clone

Candidate: `xpu-async-cloned-sampled-token-ids`

Change tested:

- Keep the XPU stream/event fix.
- Add `VLLM_XPU_ASYNC_CLONE_SAMPLED_TOKEN_IDS=1` to clone `sampler_output.sampled_token_ids` on XPU immediately after sampling. This was intended to avoid sampler-buffer lifetime or ordering issues while keeping the path GPU-local.

Outcome:

- raw145 n64 exact: pass
- raw145 n256 exact: pass
- semantic repeat suite: fail
- Failure: nondeterministic token hashes
- Observed drift matched the stream-only candidate.
- No benchmark run. No LocalMaxxing submission.

### Synchronous async-output-copy control

Control: `VLLM_XPU_SYNC_ASYNC_OUTPUT_COPY=1` with async scheduling still enabled.

Outcome:

- semantic repeat suite: fail
- Failure: nondeterministic token hashes
- Observed drift again matched the arithmetic `42` vs ` 42` case.

## Interpretation

The repeat drift survives:

- XPU-native stream/event objects
- a private GPU clone of the sampled-token tensor
- synchronous XPU output copy

That makes the CPU output-copy path unlikely to be the only root cause. The remaining suspects are upstream in async scheduling, XPU graph interaction, or decode input state ordering.

For published or shared results, async scheduling remains rejected. The safe path remains the no-async quality-valid recipe. Performance work should move back to reducing GPU graph and collective boundaries: Q/K RMS variance allreduce, hidden-state allreduce waits, MoE/projection epilogue fusion, and fewer framework-level synchronizations.

Primary data: `data/minimax-m27-async-scheduling-rejections-20260516.json`.
