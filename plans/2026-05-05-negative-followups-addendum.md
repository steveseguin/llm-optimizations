# 2026-05-05 Optimization Plan Addendum

## What Changed

- Q4_0 event-barrier allreduce remains the best quality-preserving GGUF path: 3x B70, selector `2,1,3`, `43.605 tok/s` on 512 prompt / 512 output.
- Static FP8 TP4 with vLLM/XPU FA2 and verified n-gram speculative decode remains the best four-card high-fidelity path: `46.067 tok/s` on 512 prompt / 512 output.
- The 2026-05-05 follow-up screens were negative and should not displace the current best paths.

## Negative Results To Avoid Repeating

- `GGML_SYCL_COMM_SMALL_F32=1` is a regression:
  - 4x Q4_0 `512/128`: `31.763 tok/s`, below prior 4x event-barrier `32.427 tok/s`.
  - 3x Q4_0 `512/128`: `34.874 tok/s`, far below the 3x event-barrier path.
- vLLM FP8 TP2/PP2 is not a single-stream speed path:
  - warm-cache no-spec `64/64`: `27.795 tok/s`.
  - PP2+n-gram fails with `XPUModelRunner` missing `drafter`.
- `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` hurts TP4 FP8 n-gram:
  - `512/256`: `40.049 tok/s` versus `42.245 tok/s` default.
- MiniMax `MUL_MAT_ID` split guard is diagnostic only:
  - it avoids the impossible split-buffer execution path but falls back to monolithic SYCL allocations of `20-27 GB`, which still fail.

## Next Implementation Targets

1. Q4_0 GGUF:
   - stop tuning small allreduce marker/kernel variants;
   - prototype fused output-projection `MUL_MAT` plus allreduce for the 20 KB F32 `linear_attn_out` / `attn_output` tensors;
   - alternatively prototype a reduce-scatter/all-gather style Meta scheduling path that avoids fully mirrored tiny tensors after every row-parallel projection.
2. FP8:
   - keep TP4/default oneCCL/n-gram4 lookup `2/5` as current best;
   - investigate PP2+n-gram only if a larger model requires PP2 capacity.
3. MiniMax:
   - add loader-placement diagnostics for `blk.*.ffn_*_exps.weight` buffer choices;
   - implement split-buffer `GGML_OP_MUL_MAT_ID` or expert-owned execution before more `-ncmoe` sweeps.
