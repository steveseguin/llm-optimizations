# MiniMax WS TopK Reuse Reject

Date: 2026-05-18

## Goal

Test whether only reusing the internal MiniMax WS top-k tensors is graph-safe. This was narrower than the previously rejected `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL=1` scratch attempt because it did not reuse routed intermediates or returned output buffers.

## Candidate

Added an opt-in `VLLM_XPU_MINIMAX_WS_REUSE_TOPK_ONLY=1` branch inside `moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws`:

- reuse thread-local `topk_weight [n_tokens, top_k]` as float32;
- reuse thread-local `topk_idx [n_tokens, top_k]` as int32;
- leave intermediates and output allocation unchanged.

Candidate build:

- Source SHA256: `7871f2d999eb704722814615e6184aef3d04243c022926a247e102c791d3c55e`
- `moe_int4_ops` SHA256: `45c91bfd268d33ac9855151a509e6b995b760eaf523f106828070c0da19e886e`
- Build log: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T131235Z.log`

## Outcome

The candidate failed the first raw145 n64 exact gate:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/ws-reuse-topk-only-raw145-n64-20260518T131500Z.json`
- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/ws-reuse-topk-only-raw145-n64-20260518T131500Z.log`
- Observed combined token hash: `242152df6909e5e25433f43875de5e51c210d146a22279611852b695bcf7d978`
- Expected combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- NUL token count: `63`
- Control nonspace chars: `63`
- Decision: reject without benchmark.

## Restore Check

The patch was reverted and the extension rebuilt.

- Restored source SHA256: `8d7da6669eecd1d0d0a36cdfca8e07b13620ab0c4d7c20f6dc3967f1b78b3ea5`
- Restored `moe_int4_ops` SHA256: `ab93a9b7a2c2c207c834d3ac398cc82015dac9cd1682c2b37d1212be93498a2e`
- Rebuild log: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T132053Z.log`
- Default raw145 n64 exact JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/clean-after-topk-reject-raw145-n64-20260518T132255Z.json`
- Restored default result: pass, hash `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`, no NUL/control output.

## Decision

Do not use static top-k reuse inside the MiniMax WS graph path. Even top-k-only reuse corrupts graph replay output on this stack. Future allocation work should use graph-owned buffers or explicit per-graph lifetime management rather than static thread-local tensors.
