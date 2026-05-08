# 2026-05-08 MiniMax RPC Device Mapping Fix

## Summary

The current valid MiniMax result is a corrected `4x B70` RPC layer-mode baseline:

- Model: `unsloth/MiniMax-M2.7-GGUF`, `UD-IQ4_XS`
- Engine: `ik_llama.cpp` commit `9a26522` plus local B70/SYCL patches
- Hardware: `4x Intel Arc Pro B70 32GB`, Ubuntu 24.04.4 LTS
- Corrected best: `14.292387 tok/s`, `p0/n64`, three repeats
- Samples: `13.8745`, `14.4737`, `14.529` tok/s
- LocalMaxxing: queued; API returned Cloudflare `522` for both detailed and reduced POSTs

This is quality-preserving: no speculative decoding, no smart expert reduction, no expert dropping, and no power-limit changes.

## Root Cause

Layer-mode placement computes `default_layer_device` as an index into `model.devices`, but the RPC layer assignment path passed that index directly to `llama_default_buffer_type_offload()`. In an RPC-only client this can make memory accounting and actual buffer selection disagree.

Patch:

```cpp
model.buft_layer[i] = llama_default_buffer_type_offload(
    model, model.devices[model.default_layer_device[i]]);
```

and the same mapping for the output layer.

Consequence: the old command using `-ts 0/1/1/1/1` was depending on bad index behavior. With the mapping fixed, the four-RPC-device command is `-ts 1/1/1/1`. The old split string now skips a B70 and can overload another worker.

Patch file:

```text
patches/ik-llama-minimax-rpc-device-map-and-graphsplit-20260508.patch
```

## Repro

Start workers:

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
  -p 0 -n 64 -r 3 -ngl 99 -sm layer -ts 1/1/1/1 \
  -fa 0 -nkvo 1 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

## Sweep

| Config | tok/s | Notes |
| --- | ---: | --- |
| Old split string after map fix, `-ts 0/1/1/1/1` | failed | Skips a B70 and overloads another worker |
| Corrected smoke, `p0/n1`, `-ts 1/1/1/1/0` | `4.957283` | Confirms corrected mapping loads and executes |
| Corrected split, `-sas 1` | `14.034271` | Stable, but not best after fix |
| `-no-mmad 1` | `13.836317` | Slower; keep fused MMAD on |
| `-sas 0` | `14.127082` | Better single run than `-sas 1` |
| `-t 1`, `-sas 0` | `13.994563` | Slower |
| `-t 8`, `-sas 0` | `13.976345` | Slower |
| `-muge 0`, `-sas 0` | `13.826424` | Slower; keep merged up/gate enabled |
| Best corrected, `-r 3`, `-sas 0` | `14.292387` | Samples `13.8745`, `14.4737`, `14.529` |

Raw data:

```text
data/minimax-m27-rpc-device-mapfix-20260508.json
```

## Graph Split Status

The graph/expert split path now has a working smoke test but is not competitive yet:

- `p0/n1` graph split completed after isolating the final real reduce.
- `p0/n16` graph split measured `3.813849 tok/s`.
- Scheduler trace showed `503` splits/token across 5 backends.
- Naive `GGML_SCHED_ASYNC_PER_BACKEND=1` and `GGML_SCHED_ASYNC_WAVE=1` experiments crashed the client/RPC path and remain off by default.

Interpretation: the graph path is currently dominated by split/RPC submission overhead and barriers. The next real graph optimization is dependency-aware parallel scheduling or coalescing per-device MiniMax work into larger submitted subgraphs. The current naive async experiments are useful as negative data but should not be used for valid runs.

## External Notes

MiniMax's official vLLM deployment guide recommends vLLM for MiniMax-M2.7 and lists 4-GPU TP and 8-GPU expert-parallel examples, with much larger recommended VRAM than this 4x32 GB B70 box provides for the full BF16 model:

- https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/vllm_deploy_guide.md

Intel's `llm-scaler` is relevant for the Qwen branch and as an Arc Pro B70 vLLM reference. The May 2026 README lists B70 support and Qwen3.5-27B FP8/INT4 support:

- https://github.com/intel/llm-scaler

The `Lasimeri/MiniMax-M2.7-int4-AutoRound` checkpoint is worth trying later through vLLM/SGLang when storage/RAM allows. It is W4A16 INT4 with MoE gate layers kept full precision, but the model card says `iters=0`/RTN, so it is a performance candidate rather than a known quality-equivalent replacement:

- https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound

## Next

1. Retry the queued LocalMaxxing submission when the API responds.
2. Keep the corrected layer baseline as the honest reproducible MiniMax number.
3. Continue graph/expert split work, but focus on reducing the `503` split submissions or making independent per-device branches execute concurrently with correct dependencies.
4. Defer vLLM/SGLang/AutoRound MiniMax until storage and host RAM are less likely to dominate the experiment.
