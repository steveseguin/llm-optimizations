# 2026-05-07 MiniMax M2.7 Post-Reboot Diagnostics

## Scope

- Model: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Engine: patched `llama.cpp` SYCL/Level Zero build at `/home/steve/src/llama.cpp-q4-b70`
- Build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- Hardware: 4x Intel Arc Pro B70 32GB, Level Zero `1.15.37833+4`

## Why There Is No MiniMax Score Yet

No useful MiniMax throughput result has completed. The earlier runs were diagnostic smoke tests, not benchmark submissions.

The main command shape used `-ncmoe 60`. In llama.cpp this means "keep the first 60 MoE layers on CPU." That is useful to fit dense block-0 diagnostics into one active GPU, but it cannot produce a valid all-GPU MiniMax score because the first expert op falls back to CPU.

## Current Findings

Four-card row split with `-ncmoe 60` still stalls before a one-token result in the first dense attention projection:

```text
dst='Qcur-0' src0='blk.0.attn_q.weight' type=q8_0 src1='attn_norm-0' type=f32
```

One-active row split with `-ts 1/0/0/0 -ncmoe 60` proved the q8 attention matvec can complete when only SYCL0 owns the split. That run then failed at the MiniMax f32 gate projection:

```text
dst='ffn_moe_logits-0'
src0='blk.0.ffn_gate_inp.weight' type=f32 ne=[3072,256,1,1]
src1='ffn_norm-0' type=f32 ne=[3072,1,1,1]
```

The failure was a oneAPI MKL f32 GEMM `UR_RESULT_ERROR_DEVICE_LOST`.

## Patch Applied

The existing `GGML_SYCL_SMALL_F32_MMV=1` route only handled up to 64 output rows. MiniMax's `ffn_gate_inp.weight` has 256 rows, so it still went through oneAPI MKL GEMM.

Patch:

```text
patches/llama-cpp-sycl-small-f32-mmv-threshold-20260507.patch
```

Behavior:

```bash
GGML_SYCL_SMALL_F32_MMV=1    # previous behavior, 64-row threshold
GGML_SYCL_SMALL_F32_MMV=256  # route MiniMax ffn_gate_inp through custom f32 matvec
```

With `GGML_SYCL_SMALL_F32_MMV=256`, the one-active run passed `ffn_gate_inp` and reached the first expert `MUL_MAT_ID` split. The crash then moved to the CPU fallback input sync for the first expert, caused by `-ncmoe 60`.

Evidence log:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-row-split-oneactive-ngl99-ncmoe60-smallf32-256-p0n1-20260507T094627Z.log
```

Key line:

```text
ggml_backend_sched_compute_trace: split_begin id=2 backend=4/CPU n_inputs=2 n_nodes=4 first=ffn_moe_gate-0/MUL_MAT_ID last=ffn_moe_down-0/MUL_MAT_ID
```

## Next Test

I tried a true four-card all-GPU smoke without `-ncmoe`:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZES_ENABLE_SYSMAN=1 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_SMALL_F32_MMV=256 \
GGML_SCHED_COMPUTE_TRACE=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT_DEBUG=1 \
LLAMA_EXPERT_PLACEMENT_DEBUG=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -sm row -ts 1/1/1/1 \
  -p 0 -n 1 -r 1 -ngl 99 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 --poll 50 --no-warmup -o jsonl
```

If that fits and reaches block 0, the next failure point should distinguish:

- q8 attention still fails only in real four-device split, or
- split `MUL_MAT_ID` helper is now the next blocker, or
- memory pressure prevents all-GPU MiniMax on 4x32GB in this backend.

This did not fit in the current llama.cpp/SYCL split-buffer layout. It failed during tensor upload before any token generation:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-allgpu-smallf32-256-p0n1-20260507T094947Z.log
```

Key error:

```text
ggml_backend_sycl_split_buffer_init_tensor: can't allocate 160432128 Bytes of memory on device 1 for tensor blk.12.ffn_down_exps.weight type iq4_xs rows [196608, 393216) split rows 196608 original size 160432128
```

The important positive finding is that, without `-ncmoe`, expert placement chooses SYCL split buffers instead of CPU:

```text
expert_placement: tensor=blk.0.ffn_gate_exps.weight selected_buft=SYCL0_Split
```

Next diagnostic: force later expert layers to CPU with `-ot` while keeping early experts on GPU, then intentionally abort on the first split `MUL_MAT_ID` call:

```bash
GGML_SYCL_MUL_MAT_ID_SPLIT_ABORT_AFTER_CALL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ot 'blk\.(1[2-9]|[2-5][0-9]|6[01])\.(ffn_gate_exps|ffn_down_exps|ffn_up_exps)\.weight=CPU' \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -sm row -ts 1/1/1/1 \
  -p 0 -n 1 -r 1 -ngl 99
```

That run is not a benchmark either. It is only meant to answer whether four-card q8 attention reaches the first split expert call when block-0 experts are on GPU.

The first version of that diagnostic kept expert blocks 0-11 on GPU and moved expert blocks 12-61 to CPU. It still exceeded VRAM, this time while allocating a dense attention tensor:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-gpu-experts-0-11-abort-mmid-20260507T095153Z.log
```

Key error:

```text
ggml_backend_sycl_split_buffer_init_tensor: can't allocate 5013504 Bytes of memory on device 0 for tensor blk.31.attn_q.weight type q8_0 rows [0, 1536) split rows 1536 original size 5013504
```

The next tighter diagnostic keeps only block 0 on GPU:

```bash
-ot 'blk\.([1-9]|[1-5][0-9]|6[01])\..*=CPU'
```

This should fit and stop before layer 1 if the first block reaches split `MUL_MAT_ID`.

Result: it fit and reached block 0, but did not reach split `MUL_MAT_ID`.

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-block0gpu-abort-mmid-20260507T095333Z.log
```

The run stopped at the same four-card q8 matvec point:

```text
[SYCL][OP] call ggml_sycl_mul_mat: dst='Qcur-0' src0='blk.0.attn_q.weight':type=q8_0 src1='attn_norm-0':type=f32
[SYCL][OP] call ggml_sycl_op_mul_mat/quantize_row_q8_1_sycl done
```

No `split_helper_dispatch` was printed, and no expert `MUL_MAT_ID` ran. This confirms the MiniMax blocker is still q8 split matvec under four-device row split, not the expert helper.

Next code change: add `GGML_SYCL_Q8_SPLIT_DEBUG` and `GGML_SYCL_Q8_SPLIT_SYNC` around q8 split matvec dispatch to identify the exact device/stage that stalls.

## LocalMaxxing

Not submitted. These are failure localization runs and no valid MiniMax tok/s metric exists yet.

## Later 2026-05-07 Update: Why There Is Still No MiniMax Score

There is still no valid MiniMax throughput score. The successful MiniMax runs so far are only smoke tests with almost all layers on CPU, so they are not useful leaderboard data.

New blocker summary:

- Row split reaches block 0 but crashes on MiniMax Q8_0 attention matvec before any expert dispatch.
- Layer split avoids split-buffer Q8, but llama.cpp allocates one large model buffer per device. With 4 visible Level Zero GPUs, those large multi-device allocations fail before generation.
- The `llama-bench -ot` syntax is different from the common CLI: comma creates multiple benchmark variants, while semicolon combines tensor overrides in one benchmark. Earlier comma-separated override attempts did not apply the CPU and SYCL overrides together.

Important row-split evidence:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-two-row-block0gpu-semiot-q8attn-sycl0-debug-nograph-20260507T104610Z.log
```

The corrected semicolon `-ot` command forced block-0 Q/K/V/O attention weights onto plain `SYCL0` and CPU-offloaded blocks 1-61:

```bash
-ot 'blk\.0\.attn_(q|k|v|output)\.weight=SYCL0;blk\.([1-9]|[1-5][0-9]|6[01])\..*=CPU'
```

It still lost the device immediately after submitting:

```text
dst='Qcur-0' src0='blk.0.attn_q.weight' type=q8_0 ne=[3072,6144,1,1]
level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
Error OP MUL_MAT
```

This means the row-split scheduler path is still unsafe for MiniMax Q8_0 even when the Q8 tensor is not in a split buffer. The next row-split workaround needs a real code path, not just CLI tensor overrides: route MiniMax Q8 attention through a known-good simple-buffer path or a CPU fallback, then let split expert tensors proceed.

Layer-split load evidence:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-allgpu-smoke-verbose-20260507T104935Z.log
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ngl60-smoke-p0n1-20260507T105243Z.log
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ngl56-smoke-p0n1-20260507T105401Z.log
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ngl48-smoke-p0n1-20260507T105518Z.log
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ngl36-smoke-p0n1-20260507T105934Z.log
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ngl32-smoke-p0n1-20260507T110328Z.log
```

Observed allocation failures:

```text
-ngl 99: SYCL0 allocation 27,676,983,296 bytes failed
-ngl 60: SYCL0 allocation 25,947,171,840 bytes failed
-ngl 56: SYCL0 allocation 24,217,360,384 bytes failed
-ngl 48: SYCL0 allocation 20,757,737,472 bytes failed
-ngl 36: SYCL1 allocation 15,568,303,104 bytes failed
-ngl 32: SYCL1 allocation 13,838,491,648 bytes failed
```

The standalone `tools/sycl_alloc_probe.cpp` reproduces the allocation behavior outside llama.cpp:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
icpx -fsycl -O2 tools/sycl_alloc_probe.cpp -o /tmp/sycl_alloc_probe

ONEAPI_DEVICE_SELECTOR=level_zero:0 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
/tmp/sycl_alloc_probe 0 24

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
/tmp/sycl_alloc_probe all 8
```

Observed:

```text
single visible B70: 24 GiB allocation OK
four visible B70s: 8 GiB per-GPU simultaneous allocation failed at GPU 1
```

This explains why Qwen row-split can run while MiniMax layer-split cannot: row split allocates many smaller split-buffer tensor shards, while layer split asks llama.cpp/ggml for one large per-device model buffer.

## Current Next Steps

1. Stop treating layer split as the primary MiniMax path until the large multi-device allocation issue is solved.
2. Continue the row-split MiniMax route because it uses smaller per-tensor allocations.
3. Add a focused Q8 attention fallback for MiniMax row split:
   - first option: route Q8 attention matvec to CPU only, to get a valid but slow baseline score;
   - second option: create a normal per-device temporary buffer for the Q8 shard and call the known-good non-split Q8 kernel;
   - third option: investigate the Q8_0 reordered MMVQ kernel for the `3072 x 6144` MiniMax attention shape under row-split scheduling.
4. Do not submit LocalMaxxing until a real MiniMax generation run completes.
