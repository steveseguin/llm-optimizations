# 2026-05-07 MiniMax SAS1 and OpenCL Allocator Follow-Up

## Result

- Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`
- Local path: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Engine: `ik_llama.cpp` commit `9a26522` plus local B70/SYCL patches
- Hardware: 4x Intel Arc Pro B70 32GB, Ubuntu 24.04.4 LTS
- New best: `14.425213 tok/s`, `p0/n64`, `-sas 1`, LocalMaxxing `cmow2dknq00k2p101x14ylh33`

This is a small but valid improvement over the previous `14.146009 tok/s` submitted result. It is quality-preserving: no speculative decoding, no smart expert reduction, no expert dropping, and no power-limit changes.

## Repro

Start detached RPC workers:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

for i in 0 1 2 3; do
  port=$((50100+i))
  setsid -f env ONEAPI_DEVICE_SELECTOR="level_zero:${i}" \
    UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
    ZES_ENABLE_SYSMAN=1 \
    GGML_SYCL_DISABLE_DNN=1 \
    GGML_DISABLE_FUSED_RMS_NORM=1 \
    GGML_DISABLE_FUSED_MUL_UNARY=1 \
    /home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/bin/rpc-server \
      --device SYCL0 --host 127.0.0.1 --port "$port" \
      >"/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/rpc-b70-${i}.log" 2>&1
done
```

Run benchmark:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_DISABLE_FUSED_MUL_UNARY=1 \
/home/steve/src/ik_llama.cpp/build-rpc-client-cpu/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 1 -ngl 99 -sm layer -ts 0/1/1/1/1 \
  -fa 0 -nkvo 1 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 1 -o json
```

## Micro-Sweep

| Config | tok/s | Notes |
| --- | ---: | --- |
| RPC smoke, `p0/n1` | `4.628997` | Fresh load/repack smoke after restarting detached workers |
| Baseline, `p0/n64` | `14.385361` | Same flags as previous best, without `-sas` |
| `-sas 1`, `p0/n64` | `14.425213` | New best, submitted to LocalMaxxing |
| `-sas 1 -no-ooae 1`, `p0/n64` | `14.397214` | Slight regression; keep `-no-ooae` off |

Result data:

```text
data/minimax-m27-ikrpc-sas1-and-opencl-20260507.json
```

## Direct Graph Split Findings

I added split-buffer telemetry to `ggml/src/ggml-sycl.cpp`:

- `GGML_SYCL_SPLIT_POOL_DEBUG=1`
- `GGML_SYCL_SPLIT_ALLOC_EXACT=1`
- `GGML_SYCL_SPLIT_COPY_DEBUG=1`

The Level Zero direct graph path is still blocked by single-process aggregate allocation behavior. With `--n-cpu-moe 58`, output and embeddings forced to CPU, and `GGML_SYCL_SPLIT_POOL_MIB=5120`, device 3 consumed about `5308 MiB` before failing on a tiny allocation:

```text
tensor=blk.29.attn_output.weight.0 size=2088960
pool_used=5565960192 pool_size=5565960192
```

The key surprise is that graph mode materializes suffixed dense tensors such as `blk.28.attn_q.weight.0` through `.3`, and the split allocator accounts them poorly versus the earlier estimated `3520.43 MiB` per device.

The standalone allocator probe confirmed the Level Zero ceiling:

| Backend | All-four allocation result |
| --- | --- |
| Level Zero | `5.10 GiB` per GPU fails at GPU 2 |
| OpenCL | `24 GiB` per GPU succeeds |

OpenCL is not currently a practical MiniMax answer on this 16 GB RAM host. It can reserve large device allocations, but all-GPU graph upload hit system OOM, and the reduced `--n-cpu-moe 58` run consumed heavy swap and made no progress past startup before I stopped it. This points to OpenCL host-memory mapping pressure rather than a clean model execution path.

## Next

1. Keep RPC+Level Zero as the only valid MiniMax path for public numbers.
2. Treat `-sas 1` as the current default for MiniMax RPC layer runs.
3. Do not use `-no-ooae 1` for this path.
4. For >30 tok/s, the next real work is still code-level expert/tensor parallelism: either an RPC split-buffer implementation or a direct graph split allocator that avoids Level Zero single-process aggregate mapping limits.

## Candidate Non-GGUF Path: Lasimeri AutoRound INT4

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Initial read:

- Format is safetensors, not GGUF; repo size is about 121 GB.
- Quantization is W4A16, group size 128, symmetric INT4 weights with FP16/BF16 activations.
- The model card says the MoE gate layers are left full precision.
- The quantization config says `iters=0`, so this is RTN-style AutoRound rather than a fully optimized iterative calibration run.
- Suggested runtime is vLLM or SGLang with tensor parallelism; neither stack is installed on this host yet.

Assessment:

- Worth testing because vLLM has native MiniMax-M2 and tensor/expert-parallel execution paths, which is closer to the architecture we need than llama.cpp RPC layer splitting.
- Not a proven quality upgrade over `UD-IQ4_XS`; `iters=0` is a quality caveat.
- Not a drop-in replacement for the current GGUF path; it requires installing a PyTorch XPU/vLLM or SGLang XPU stack and likely downloading to the external 4 TB drive or freeing internal storage.
- It may be too tight for 4x32 GB at long context because the checkpoint is 121 GB before runtime/KV/cache overhead, but a short-context load test is still valuable.
