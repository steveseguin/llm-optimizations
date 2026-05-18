# MiniMax MoE WS Tile Knobs Patch

Date: 2026-05-18

## Purpose

This local patch added diagnostic-only tile selection knobs to the llm-scaler INT4 MoE work-sharing path:

- `VLLM_XPU_MOE_WS_UP_NTILE=2|4|8`
- `VLLM_XPU_MOE_WS_DOWN_HTILE=4|8`

The intent was to test whether fewer decode work-items improved B70 single-session decode while preserving exact MiniMax routing and output quality.

## Source

Patched file:

- `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`

Resulting hashes:

- Source SHA256: `7064e92719c598a12d0727bc71a9d134dac166f0a9b77ea6f04c06bf50039c3e`
- Built extension SHA256: `5d6e85788590adc769a25b0c2606266fec68b92856a0384f1756a73ea261483c`

Build command:

```bash
MAX_JOBS=2 /home/steve/llm-optimizations-publish/scripts/build-llm-scaler-moe-int4-xpu.sh
```

Build log:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-oneapi2025-20260518T112129Z.log`

## Code Shape

The patch added:

- `parse_env_choice(...)`
- `forced_ws_up_ntile()`
- `forced_ws_down_htile()`

Then it routed `moe_ws_up_cutlass_int4_kernel(...)` and `moe_ws_up_cutlass_int4_with_shared_fp16_kernel(...)` through the forced `N_TILE` when set, otherwise preserving the original policy:

- `n_tokens <= 2`: `N_TILE=2`
- `n_tokens <= 4`: `N_TILE=4`
- else: `N_TILE=8`

It also routed the down kernel through forced `H_TILE=4` or `H_TILE=8` when set, otherwise preserving the original policy:

- `n_tokens <= 2`: `H_TILE=4`
- else: `H_TILE=8`

The strict runner was updated to record both env fields in `candidate_env`.

## Result

Do not promote this patch as a speed win:

- `UP_NTILE=4`: strict quality passed but slowed to `79.236469` output tok/s.
- `UP_NTILE=8`: graph-mode raw canary stalled after compile.
- `DOWN_HTILE=8`: raw canary produced a different exact token hash.

The patch is useful as a reproducible negative experiment and as a temporary diagnostic knob, not as a production default.
