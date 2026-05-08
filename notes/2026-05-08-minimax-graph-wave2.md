# 2026-05-08 MiniMax Graph Wave2

## Result

The safer graph-wave scheduler experiment now runs without the earlier malformed RPC response crash:

- Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`
- Engine: `ik_llama.cpp` commit `9a26522` plus local RPC/SYCL patches
- Hardware: `4x Intel Arc Pro B70 32GB`
- Graph baseline before wave2: `3.813849 tok/s`, `p0/n16`
- Wave2 graph: `3.920533 tok/s`, `p0/n16`
- Improvement: about `2.8%`

This is not a recommended runtime path yet. The corrected layer-mode baseline is still `14.292387 tok/s`, so graph mode is currently a diagnostic path for expert/tensor-split development.

## Fixes

I hardened the earlier unsafe wave experiment:

- Added a per-socket mutex around full RPC command/response transactions.
- Added a mutex around the RPC graph cache.
- Changed `GGML_SCHED_ASYNC_WAVE=1` so input copies remain ordered and main-thread-owned.
- Wave grouping now requires distinct target backends and no input produced by another split in the same wave.
- Completion bookkeeping runs after wave workers join.
- Renamed the older per-backend branch to `GGML_SCHED_ASYNC_PER_BACKEND_UNSAFE` so it is not used accidentally.

Patch snapshot:

```text
patches/ik-llama-minimax-rpc-device-map-and-graphsplit-20260508.patch
```

## Repro

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
GGML_SCHED_ASYNC_WAVE=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 16 -r 1 -ngl 99 -sm graph -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 1 -o json
```

## Findings

Scheduler trace still shows `503` splits/token:

```text
backend[0] RPC B70-0: 125 splits
backend[1] RPC B70-1: 125 splits
backend[2] RPC B70-2: 125 splits
backend[3] RPC B70-3: 127 splits
backend[4] CPU: 1 split
```

The graph repeats a pattern where one backend computes a small normalization/setup split and the other branches depend on that output before they can run. Wave2 can parallelize some independent branch work, but it cannot remove the many RPC graph submissions or the cross-backend setup dependency.

Flash attention was tested and is currently not usable in this path:

```text
ggml_backend_sycl_graph_compute: error: op not supported fa-0 (FLASH_ATTN_EXT)
```

## Next

1. Keep graph wave behind `GGML_SCHED_ASYNC_WAVE=1`; do not publish it as a recommended result.
2. Change MiniMax graph construction so cheap norm/elementwise setup is replicated per branch instead of produced by one backend and copied to the other branches.
3. Continue looking for ways to reduce split submissions before deeper parallel scheduling work.
4. Keep layer mode as the public MiniMax number until graph mode is at least competitive with `14.292387 tok/s`.
