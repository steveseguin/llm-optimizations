# 2026-05-07 MiniMax ik_llama.cpp RPC+SYCL Baseline

## Result

- Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`, local shard entry `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Engine: `ik_llama.cpp` commit `9a26522` plus local SYCL/RPC patches
- Hardware: 4x Intel Arc Pro B70 32GB, Ubuntu 24.04.4 LTS
- First valid result: `13.754201 tok/s`, `p0/n64`, batch 1, LocalMaxxing `cmovvoo6f00f5p1017yeb7kxd`
- Current best after the IQ4_XS merged up/gate active-expert fast path: `14.146009 tok/s`, `p0/n64`, batch 1, warm worker process, LocalMaxxing `cmovzx6um00hrp101ldegbaga`

This is the first useful MiniMax four-B70 throughput baseline. It uses one SYCL RPC worker process per GPU to avoid the single-process Level Zero/SYCL aggregate allocation ceiling observed with the 101 GiB GGUF.

The first version kept fused MoE disabled. The current patch also adds a conservative SYCL `MOE_FUSED_UP_GATE` implementation, which internally reuses SYCL `MUL_MAT_ID` and then runs a gate activation/multiply kernel. That makes `-fmoe 1` work and allows `-muge 1` merged gate/up experts, but it is still not the low-level fused expert kernel needed for the >30 tok/s target.

The latest patch adds a more direct IQ4_XS merged up/gate path for active experts. It quantizes the current token activations once to `q8_1`, reads only the selected experts, computes gate and up dot-products in one SYCL kernel, applies the gate activation, and writes the gated expert output. This is quality-preserving for the tested path. The gain is modest, about 1.1% over the conservative fused-MoE path, but it confirms that the active-expert path can be specialized safely. An attempted matching IQ4_XS `MUL_MAT_ID` down-projection fast path regressed to `13.207767 tok/s`, so it is now opt-in only via `GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1`.

## Launch

Start four workers:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
OUTDIR=/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/rpc-live
mkdir -p "$OUTDIR"

for i in 0 1 2 3; do
  port=$((50100+i))
  log="$OUTDIR/rpc-b70-${i}.log"
  ONEAPI_DEVICE_SELECTOR="level_zero:${i}" \
  UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
  ZES_ENABLE_SYSMAN=1 \
  GGML_SYCL_DISABLE_DNN=1 \
  GGML_DISABLE_FUSED_RMS_NORM=1 \
  GGML_DISABLE_FUSED_MUL_UNARY=1 \
  /home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/bin/rpc-server \
    --device SYCL0 --host 127.0.0.1 --port "$port" >"$log" 2>&1 &
done
```

Run the benchmark:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 1 -ngl 99 -sm layer -ts 0/1/1/1/1 \
  -fa 0 -nkvo 1 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -v -o json
```

Result JSON:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/ik-rpc-quad-layer-ts0-1111-muge1-fmoe1-fastiq4xsmoe-default-warm-rtr1-t4-ub32-p0n64-nkvo1-20260507T211315Z.jsonl
```

## Micro-Sweep

| Config | tok/s | Notes |
| --- | ---: | --- |
| `p0/n16`, CPU KV, no runtime repack | `9.977498` | First stable multi-token RPC+SYCL result after `MULTI_ADD` |
| `p0/n64`, GPU KV | `11.353396` | Fewer graph splits, but slower |
| `p0/n64`, CPU KV, no runtime repack, `-t 8` | `12.564798` | Stable baseline |
| `p0/n64`, CPU KV, runtime repack, `-t 8` | `13.465415` | Repack helps |
| `p0/n64`, CPU KV, runtime repack, `-t 16` | `12.826655` | Too many CPU threads for this path |
| `p0/n64`, CPU KV, runtime repack, `-t 4` | `13.754201` | Best |
| `p0/n64`, CPU KV, runtime repack, `-t 2` | `13.671846` | Slightly below `-t 4` |
| `p0/n64`, CPU KV, runtime repack, `-t 4`, `-ub 64` | `13.690724` | Slightly below `-ub 32` |
| `p0/n64`, CPU KV, runtime repack, `-t 4`, `-ub 16` | `13.650250` | Slightly below `-ub 32` |
| Experimental fused `MUL_MULTI_ADD`, `-t 4`, `-ub 32` | `12.330226` | Generic SYCL fused MMAD kernel is slower than decomposed `mul + MULTI_ADD` |
| SYCL `MOE_FUSED_UP_GATE`, `-fmoe 1 -no-mmad 1`, `-t 4`, `-ub 32` | `13.839857` | Fused up-gate now executes; small improvement |
| SYCL `MOE_FUSED_UP_GATE`, `-fmoe 1 -no-mmad 0`, `-t 4`, `-ub 32` | `13.899895` | Fused MMAD is no longer a regression when paired with fused up-gate |
| SYCL `MOE_FUSED_UP_GATE`, `-muge 1 -fmoe 1 -no-mmad 0`, `-t 4`, `-ub 32` | `13.989985` | Conservative fused-MoE best; LocalMaxxing `cmovxb67400g6p10100by6frn` |
| Timing probe, warm `p0/n1`, conservative fused-MoE | `5.101027` | `MOE_FUSED_UP_GATE` was about 34 ms across 62 calls; `MUL_MAT_ID` down path about 10 ms; router matmuls had first-touch spikes but steady calls were tens of microseconds |
| `-sm graph -ts 0/1/1/1/1 --fit 1` | fail | Split vector included CPU as a nonzero split target; auto-fit tried to place output on CPU device 4 with 0 MiB free |
| `-sm graph -ts 1/1/1/1/0 --fit 1`, `-b 32 -ub 32` | fail | Corrected split reached buffer planning but a worker tried a single `33904507904` byte SYCL allocation on one B70; removing `-muge` did not fix it |
| IQ4_XS merged up/gate active-expert fast path, `p0/n64` | `14.130936` | First full run after adding the specialized active-expert kernel |
| IQ4_XS down `MUL_MAT_ID` fast path, opt-in | `13.207767` | Regression; disabled by default behind `GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1` |
| IQ4_XS merged up/gate active-expert fast path, warm worker, `p0/n64` | `14.146009` | Current best; no quality-sacrificing flags, no power changes; LocalMaxxing `cmovzx6um00hrp101ldegbaga` |

## Patches

Patch artifact:

```text
patches/ik-llama-minimax-rpc-sycl-20260507.patch
```

The patch includes:

- `llama-bench -no-mmad` to disable fused `MUL_MULTI_ADD` from the benchmark CLI.
- `GGML_DISABLE_FUSED_RMS_NORM` and `GGML_DISABLE_FUSED_MUL_UNARY` env fallbacks to decompose fused graph ops that the SYCL RPC worker does not support yet.
- SYCL `SIGMOID`, needed by the decomposed MiniMax MoE path.
- SYCL `MULTI_ADD`, which makes the decomposed expert combine path run on the B70 instead of failing backend support.
- Experimental SYCL `MUL_MULTI_ADD`, kept because it executes correctly enough to benchmark, but currently regresses speed.
- SYCL `MOE_FUSED_UP_GATE`, currently implemented as an internal `MUL_MAT_ID` decomposition plus a fused gate activation/multiply kernel.
- A direct IQ4_XS merged up/gate active-expert path for `MOE_FUSED_UP_GATE`; this is enabled by default for the matching merged expert layout.
- An experimental IQ4_XS `MUL_MAT_ID` down-projection path; this is disabled by default after benchmarking slower than the existing path.
- `GGML_SYCL_OP_TIMING=1`, an opt-in per-op timing probe for SYCL graph execution.

## Current Blockers

- `-fa 1` reaches `FLASH_ATTN_EXT`, which is unsupported in the SYCL RPC worker and crashes.
- `-sm graph` is still blocked. With an incorrect `0/1/1/1/1` split, auto-fit attempts to use the CPU as a split device. With the corrected `1/1/1/1/0` split, one RPC worker attempts a single about-33.9 GB SYCL allocation and fails even with `-b 32 -ub 32` and even without `-muge`.
- A combined `-t 2,4,8` llama-bench sweep with `-muge 1` stopped making progress during `llama_repack_up_gate_exps` around layer 56, while the single `-t 4` run completed. Treat thread-list sweeps with merged gate/up repack as unstable for now.
- The current layer-split RPC path mostly turns the four B70s into memory shelves. To reach >30 tok/s, MiniMax likely needs viable graph/tensor split or a much more fused expert pipeline that keeps per-layer active expert work parallel across cards.

## Next Work

1. Investigate graph split buffer planning and RPC allocation behavior. The immediate question is why corrected graph mode requests a single near-VRAM-sized allocation on one worker instead of using smaller buffers.
2. Profile the new IQ4_XS up/gate path with `GGML_SYCL_OP_TIMING=1` and compare per-layer timings against the conservative path; keep the path only if it remains neutral or positive after warm-up.
3. Rework the down-projection fast path before enabling it again. The naive active-expert IQ4_XS down kernel is slower than the existing `MUL_MAT_ID` path.
4. Look for a higher-level parallelism path for MiniMax, including tensor/graph split fixes and external references such as multi-GPU MiniMax deployments on CUDA/ROCm stacks.
5. Keep process-per-GPU RPC as the MiniMax path until the single-process Level Zero aggregate allocation limit is understood or worked around.
