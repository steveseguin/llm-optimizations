# 2026-05-07 MiniMax ik_llama.cpp RPC+SYCL Baseline

## Result

- Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`, local shard entry `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Engine: `ik_llama.cpp` commit `9a26522` plus local SYCL/RPC patches
- Hardware: 4x Intel Arc Pro B70 32GB, Ubuntu 24.04.4 LTS
- First valid result: `13.754201 tok/s`, `p0/n64`, batch 1, LocalMaxxing `cmovvoo6f00f5p1017yeb7kxd`
- Current best after the SYCL fused-MoE unlock: `13.989985 tok/s`, `p0/n64`, batch 1, LocalMaxxing `cmovxb67400g6p10100by6frn`

This is the first useful MiniMax four-B70 throughput baseline. It uses one SYCL RPC worker process per GPU to avoid the single-process Level Zero/SYCL aggregate allocation ceiling observed with the 101 GiB GGUF.

The first version kept fused MoE disabled. The current patch also adds a conservative SYCL `MOE_FUSED_UP_GATE` implementation, which internally reuses SYCL `MUL_MAT_ID` and then runs a gate activation/multiply kernel. That makes `-fmoe 1` work and allows `-muge 1` merged gate/up experts, but it is still not the low-level fused expert kernel needed for the >30 tok/s target.

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
  -rtr 1 -fmoe 0 -no-mmad 1 -v -o json
```

Result JSON:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/ik-rpc-quad-layer-ts0-1111-nofmoe-nommad-syclmultiadd-rtr1-t4-p0n64-nkvo1-20260507T185647Z.jsonl
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
| SYCL `MOE_FUSED_UP_GATE`, `-muge 1 -fmoe 1 -no-mmad 0`, `-t 4`, `-ub 32` | `13.989985` | Current best; merged gate/up expert tensors reduce one source of up/gate overhead |

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

## Current Blockers

- `-fa 1` reaches `FLASH_ATTN_EXT`, which is unsupported in the SYCL RPC worker and crashes.
- `-sm graph` attempts to allocate about 34 GB on one B70 and fails, so graph split is not viable for this GGUF layout yet.
- A combined `-t 2,4,8` llama-bench sweep with `-muge 1` stopped making progress during `llama_repack_up_gate_exps` around layer 56, while the single `-t 4` run completed. Treat thread-list sweeps with merged gate/up repack as unstable for now.
- The biggest speed opportunity is replacing the current conservative SYCL `MOE_FUSED_UP_GATE` implementation with a layout-aware fused active-expert matvec/combine kernel.

## Next Work

1. Replace the conservative SYCL `MOE_FUSED_UP_GATE` decomposition with a lower-level active-expert kernel that avoids two separate `MUL_MAT_ID` passes and repeated temp buffers.
2. Make the fused `MUL_MULTI_ADD` kernel use a layout-aware reduction strategy instead of one scalar loop per output element.
3. Recheck `-muge 1`, runtime repack, and fused MMAD once the fused expert kernel exists; target is to keep the full expert path on GPU and move toward 30 tok/s.
4. Keep process-per-GPU RPC as the MiniMax path until the single-process Level Zero aggregate allocation limit is understood or worked around.
