# MiniMax M2.7 Split `MUL_MAT_ID` TG Prototype

Date: 2026-05-05

## Context

Earlier MiniMax M2.7 UD-IQ4_XS four-B70 attempts showed that row split could place ordinary tensors across the cards, but expert tensors using `GGML_OP_MUL_MAT_ID` were either rejected for SYCL split buffers or fell back to monolithic placement. The monolithic fallback produced a large SYCL3 allocation, for example `20157825024` bytes at `-ncmoe 50`.

The new experiment adds an env-gated prototype for token-generation shape split expert execution:

```bash
GGML_SYCL_MUL_MAT_ID_SPLIT=1
```

The patch is stored at:

`/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-minimax-split-mulmatid-tg.patch`

A smaller focused patch for the new split `MUL_MAT_ID` pieces, intended to be applied on top of the existing experimental B70 llama.cpp tree, is stored at:

`/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-minimax-split-mulmatid-tg-focused.patch`

The GitHub artifact repo stores this focused patch as base64 at `patches/llama-cpp-sycl-minimax-split-mulmatid-tg-focused.patch.b64`. Decode with:

```bash
base64 -d patches/llama-cpp-sycl-minimax-split-mulmatid-tg-focused.patch.b64 > llama-cpp-sycl-minimax-split-mulmatid-tg-focused.patch
```

## Implementation Shape

The prototype is intentionally narrow:

- handles SYCL split-buffer `src0` for `GGML_OP_MUL_MAT_ID`;
- targets the token-generation case where `src1` has one token (`ne12 == 1`);
- maps selected expert IDs to flattened row ranges;
- intersects each expert row range with each B70's split-buffer row shard;
- runs existing per-device `ggml_sycl_mul_mat` on fake non-split row views;
- copies each partial row result back into the original destination tensor.

This is a correctness/viability probe, not a tuned path. It allocates temporary per-device `src1` and `dst` buffers per expert shard and does not yet batch selected experts.

## Build

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j 2
```

The build completed successfully with the existing AOT SYCL configuration.

## Test Commands

Representative command shape:

```bash
source /opt/intel/oneapi/setvars.sh --force

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZES_ENABLE_SYSMAN=1 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT=1 \
GGML_SYCL_DISABLE_DNN=1 \
LLAMA_EXPERT_PLACEMENT_DEBUG=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -v \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -sm row \
  -ts 1/1/1/1 \
  -p 0 \
  -n 1 \
  -r 1 \
  -ngl 99 \
  -ncmoe 60 \
  -fa 1 \
  -ub 32 \
  -ctk f16 \
  -ctv f16 \
  -t 8 \
  --poll 50 \
  --no-warmup \
  -o jsonl
```

## Results So Far

`-ncmoe 50` with split `MUL_MAT_ID` enabled no longer falls back to a 20 GB monolithic SYCL3 expert buffer. The expert placement log shows the late expert layers using `SYCL3_Split`, but load still fails on a much smaller per-shard allocation:

```text
blk.60.ffn_up_exps.weight type iq3_s rows [0, 98304), 129761280 bytes on device 0
```

`-ncmoe 60` progressed further. It loaded the model, constructed context, assigned KV across all four B70s, and reserved the graph:

```text
sched_reserve: fused Gated Delta Net (autoregressive) enabled
sched_reserve: fused Gated Delta Net (chunked) enabled
sched_reserve: graph nodes  = 3975
sched_reserve: graph splits = 122
```

The first one-token decode did not complete. After more than eight minutes, `strace` showed the main thread spinning in `sched_yield()` more than one million times in a few seconds, with no new llama.cpp log output and an empty JSONL result. The run was terminated manually to free the system.

This is proof that the loader/context blocker is cleared, but the naive per-expert runtime implementation is not viable yet. The next MiniMax change should instrument the first `GGML_OP_MUL_MAT_ID` runtime call and replace the current queue/copy/wait shape before trying another full run.

## Next Work

- Add targeted diagnostics inside the split `MUL_MAT_ID` helper so the first runtime call prints selected expert count, row shards, and copy directions.
- Batch selected expert shards instead of allocating/copying per expert shard.
- Replace peer copies with the same dev-to-dev copy helper pattern already used by `ggml_sycl_op_mul_mat` if the current copy path stalls.
- Add prompt/batch support only after the token-generation path produces a correct one-token result.
- Keep `-ncmoe` sweeps after the helper is viable; flag sweeps alone will not solve this model.
