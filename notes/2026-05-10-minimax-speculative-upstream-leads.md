# MiniMax Speculative Upstream Leads, 2026-05-10

## Why This Matters

Local MiniMax M2.7 DFlash/EAGLE3 screens currently load and compile but stall in
vLLM's XPU shared-memory broadcast/scheduler path before producing tokens. Since
the new target is `75+` output tok/s with target-verified speculation, this
track remains important, but it needs a scheduler/runtime fix before it can be a
benchmark path.

## Upstream Signals

vLLM release notes mention both MiniMax-M2 Eagle3 support and a major XPU
platform overhaul:

- `https://github.com/vllm-project/vllm/releases`
- relevant release-note items: MiniMax-M2 Eagle3 support, XPU deprecating IPEX
  in favor of `vllm-xpu-kernels`, WNA16, scaled_mm, MoE, and FP8 MoE support.

The MiniMax-M2 Eagle3 support PR was merged on 2026-04-06:

- `https://github.com/vllm-project/vllm/pull/37512`
- important implementation clue: it adds `SupportsEagle3`/`EagleModelMixin`
  plumbing to MiniMaxM2 and collects auxiliary hidden states in the verifier
  forward path.

There is an open vLLM issue reporting an EAGLE3 performance regression between
v0.18 and v0.19 on H200 TP4. The reported acceptance rate was unchanged while
the speedup dropped, and the author suspected the zero-bubble async scheduling
plus spec decode changes:

- `https://github.com/vllm-project/vllm/issues/39940`

This does not prove our B70 hang has the same cause, but it makes async
scheduling/spec-decode interaction a high-priority local debugging target.

vLLM's P-EAGLE writeup is relevant for future targets because it removes
EAGLE's sequential draft-token bottleneck by generating draft tokens in one
forward pass:

- `https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-03-13-p-eagle.md`

No MiniMax P-EAGLE draft was found locally. This is a future training/download
track, not an immediate B70 screen.

The `vllm-project/speculators` repo now documents DFlash training/runtime
support and notes that DFlash models can run in vLLM:

- `https://github.com/vllm-project/speculators`

The local MiniMax-specific DFlash draft is therefore conceptually aligned with
upstream direction. The blocker is not the idea; it is our XPU runtime path.

The DFlash speculators parsing PR was merged on 2026-04-15:

- `https://github.com/vllm-project/vllm/pull/38300`
- useful reference details: DFlash config auto-detection, override handling,
  verifier/draft weight loading, DFlash metrics, and tests using Qwen3-8B
  acceptance-length measurements around `1.77` to `1.84`.

The async-scheduling/spec-decode optimization PR from 2026-02-04 is also a
candidate source reference:

- `https://github.com/vllm-project/vllm/pull/33612`
- it changed spec-token placeholder handling in async scheduling; this is not
  our exact stall, but it is in the scheduler/spec boundary we need to inspect.

The vLLM hidden-state extraction RFC explains the hidden-state plumbing used by
EAGLE3/DFlash and frames hidden-state transfer as a hotpath-sensitive problem:

- `https://github.com/vllm-project/vllm/issues/33118`

That reinforces the local suspicion that our stall is around multiprocess
hidden-state/broadcast plumbing rather than model weights alone.

## Local Follow-Ups

1. Reproduce the DFlash p64/n32 stall with a profiler or targeted logging around
   `shm_broadcast.py`, `gpu_model_runner.py` speculative proposer calls, and the
   aux-hidden-state path.
2. Try disabling or bypassing async scheduling for spec decode only if vLLM still
   exposes a supported path. Plain ngram disabled async and became slow, but
   DFlash/EAGLE may need a different scheduler split.
3. Review vLLM PRs around MiniMax-M2 Eagle3 and DFlash integration before
   patching; if upstream has fixed scheduler edge cases after the installed
   `0.20.1` snapshot, a source update may be cheaper than local surgery.
4. Track P-EAGLE-style draft models for MiniMax. If a MiniMax P-EAGLE or
   DFlash variant with better acceptance appears, it may move the realistic
   speculation target above `75` output tok/s once the XPU stall is fixed.
