# MiniMax WS Internal Scratch-Reuse Rejection

Date: 2026-05-18

## Goal

Reduce per-token MoE overhead in the current exact MiniMax router-logits WS path by reusing internal scratch tensors for:

- top-k route weights;
- top-k route indices;
- routed intermediate activations.

The returned MoE output tensor was intentionally left freshly allocated to reduce graph aliasing risk.

## Candidate

Opt-in env:

- `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL=1`

Active baseline env was otherwise unchanged:

- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- XPU PIECEWISE graph, default XPU FlashAttention v2, MBT512

## Result

Rejected immediately. The first raw145 n64 exact canary failed before any benchmark:

- Expected token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed token hash: `242152df6909e5e25433f43875de5e51c210d146a22279611852b695bcf7d978`
- Generated output was corrupt/degenerate:
  - `63` NUL tokens
  - `63` non-space control characters
  - only `2` distinct generated tokens

No benchmark was run.

## Safety Check

After the failed opt-in test, the same rebuilt extension was tested with `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL` unset. The current promoted exact logits-WS path still passed raw145 n64:

- Observed token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- NUL tokens: `0`
- Control-character output: `false`
- Degenerate output: `false`

## Interpretation

The failure is consistent with scratch-buffer aliasing across XPU graph capture/replay or cross-layer opaque-op scheduling. Even though the output tensor was fresh, reusing internal tensors inside the C++ op is not safe in the compiled graph path as implemented.

Do not enable `VLLM_XPU_MINIMAX_WS_REUSE_INTERNAL=1`. Future work on allocation overhead needs a graph-aware scratch allocator or explicit graph-safe lifetime management, not static thread-local tensor reuse.

## Artifacts

- Candidate summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ws-reuse-internal-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T105145Z-summary.json`
- Failed raw canary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ws-reuse-internal-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T105145Z-quality/raw145-n64-exact.json`
- Default-path safety canary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/default-after-ws-reuse-reject-raw145-n64-20260518T105752Z.json`
- Build log: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T104831Z.log`
