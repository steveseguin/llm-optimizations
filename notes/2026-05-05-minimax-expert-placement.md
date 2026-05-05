# 2026-05-05 MiniMax Expert Placement Diagnosis

## Patch

Added `LLAMA_EXPERT_PLACEMENT_DEBUG=1` to `src/llama-model-loader.cpp`.

For `blk.*.ffn_{gate,down,up}_exps.weight`, the loader now logs:

- op name;
- tensor type and shape;
- selected buffer type and selected device;
- preferred buffer type and preferred device;
- whether the tensor was moved away from the preferred buffer;
- mmap state.

Patch: `patches/llama-cpp-loader-expert-placement-debug.patch`.

## Result

Command shape:

```bash
LLAMA_EXPERT_PLACEMENT_DEBUG=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -p 0 -n 1 -r 1 --no-warmup -ngl 99 -ncmoe 50 \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 -sm row -ts 1/1/1/1
```

Log:

`/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe50-expert-placement-20260505T001625Z.log`

Placement records:

- `186` expert tensor placement records total.
- `150` selected `CPU`.
- `36` selected plain `SYCL3`.

Final failure:

`unable to allocate SYCL3 buffer of size 20157825024`

## Interpretation

The earlier `MUL_MAT_ID` guard correctly prevents the loader from selecting `SYCL3_Split` for an op that SYCL execution cannot support. The fallback is the problem: the remaining 12 GPU-resident expert layers are placed as complete monolithic tensors on `SYCL3`, not spread across all four B70s.

That means the `-ncmoe 50` layout asks GPU3 to hold roughly 20.16 GB of expert tensors before other model memory, which explains the allocation failure.

MiniMax is now clearly blocked on split-buffer or expert-sharded `GGML_OP_MUL_MAT_ID` execution. More `-ncmoe` flag sweeps will not create a useful all-GPU path.
