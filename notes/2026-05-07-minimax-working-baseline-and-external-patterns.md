# 2026-05-07 MiniMax M2.7 Working Baseline and External Patterns

## Scope

- Model: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- Quant: Unsloth `UD-IQ4_XS` GGUF
- Engine: patched `llama.cpp` SYCL/Level Zero build `db44417+local-sycl`
- Build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- Hardware: 4x Intel Arc Pro B70 32GB, Level Zero `1.15.37833+4`

## 2026-05-07 Update: RPC Baseline Supersedes This Layer-Split Diagnostic

The direct llama.cpp SYCL layer-split diagnostic below remains useful as a failure record, but it is no longer the best MiniMax path. The current working direction is `ik_llama.cpp` with one RPC worker process per B70, which bypasses the single-process Level Zero/SYCL aggregate allocation ceiling.

Current best:

```text
13.989985 tok/s, p0/n64, 4x B70, UD-IQ4_XS, ik_llama.cpp RPC+SYCL
LocalMaxxing: cmovxb67400g6p10100by6frn
```

Detailed repro:

```text
notes/2026-05-07-minimax-ikrpc-sycl-13tok-baseline.md
data/minimax-m27-ikrpc-sycl-13tok-baseline-20260507.json
patches/ik-llama-minimax-rpc-sycl-20260507.patch
```

## External Patterns Worth Borrowing

Official MiniMax guidance points at vLLM, not GGUF, and recommends tensor parallelism for 4 GPUs plus expert parallelism for 8 GPUs:

- MiniMax vLLM guide: <https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blame/main/docs/vllm_deploy_guide.md>
- 4-GPU command uses `--tensor-parallel-size 4`.
- 8-GPU command adds `--enable_expert_parallel --tensor-parallel-size 8`.
- The guide lists 220 GB for weights and 240 GB per 1M context tokens, which explains why the unquantized/FP8 path is a poor fit for 4x32GB B70 without a smaller quant or expert sharding.

Community examples show two broad high-performance families:

- SGLang/vLLM with NVFP4/AWQ on large-memory NVIDIA cards:
  - 2x RTX PRO 6000 Blackwell with `lukealonso/MiniMax-M2.7-NVFP4`, SGLang, modelopt FP4, TP=2, b12x MoE runner, and FlashInfer reports roughly 100+ tok/s single-session decode.
  - Source: <https://www.reddit.com/r/LocalLLaMA/comments/1sjx7kg/minimaxm27_nvfp4_on_2x_rtx_pro_6000_blackwell/>
- 8x 3090 Ti AWQ examples:
  - `cyankiwi/MiniMax-M2.7-AWQ-4bit` on 8x 3090 Ti reports about 17.6 tok/s for `tg512 (c1)`.
  - Source: <https://gist.github.com/adamo1139/372de9f6cdfd38155d0dbea0b2bb3878>
- `ik_llama.cpp` exposes graph split, active-expert-only offload, graph reduce types, and `--max-gpu` controls that mainline llama.cpp does not currently expose.
  - Source: <https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md>

These examples suggest the next useful B70 work is not more `-ncmoe` sweeping. We need either:

1. a working expert-sharded `MUL_MAT_ID` path for SYCL row split, or
2. a graph/expert placement scheduler that can distribute MoE expert weights without packing all tail experts into one device.

## Working Baseline

Layer split works if most MoE experts are CPU-mapped:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZES_ENABLE_SYSMAN=1 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_SMALL_F32_MMV=256 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -sm layer -ts 1/1/1/1 \
  -p 0 -n 64 -r 1 -ngl 99 -ncmoe 56 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 --poll 50 --no-warmup -o jsonl
```

Result:

```text
0.472124 tok/s, prompt=0, output=64
```

Log:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ncmoe56-p0n64-20260507T122246Z.log
```

LocalMaxxing:

```text
cmovgojv80008n501lupo67x6
```

This is a diagnostic baseline, not an optimized result. `-ncmoe 56` means the first 56 of 62 MoE expert layers are CPU-mapped; only the final 6 expert layers are GPU-resident, all on SYCL3.

## Layer Split Staircase

| Config | Result | Notes |
| --- | ---: | --- |
| `-ncmoe 62`, `p0/n1` | `0.229877 tok/s` | all MoE experts CPU-mapped |
| `-ncmoe 58`, `p0/n1` | `0.233237 tok/s` | final 4 expert layers on SYCL3; SYCL3 model buffer `7556.29 MiB` |
| `-ncmoe 56`, `p0/n1` | `0.236161 tok/s` | final 6 expert layers on SYCL3; SYCL3 model buffer `10760.29 MiB` |
| `-ncmoe 56`, `p0/n16` | `0.410253 tok/s` | stable longer decode |
| `-ncmoe 56`, `p0/n64` | `0.472124 tok/s` | submitted diagnostic baseline |
| `-ncmoe 54`, default optimize | abort | Q8 reorder temp allocation failed after loading a `13964.29 MiB` SYCL3 model buffer |
| `-ncmoe 54`, `GGML_SYCL_DISABLE_OPT=1`, `p0/n1` | `0.268751 tok/s` | disabling reorder avoids the abort |
| `-ncmoe 54`, `GGML_SYCL_DISABLE_OPT=1`, `-b 1 -ub 1`, `p0/n16` | `0.403948 tok/s` | stable but not faster than `-ncmoe 56` |
| `-ncmoe 54`, `GGML_SYCL_REORDER_HOST_FALLBACK=0`, `p0/n1` | `0.267557 tok/s` | targeted patch skips failed reorder temp allocations without global `GGML_SYCL_DISABLE_OPT=1` |
| `-ncmoe 54`, `GGML_SYCL_REORDER_HOST_FALLBACK=0`, `-b 1 -ub 1`, `p0/n16` | `0.389894 tok/s` | stable, slightly slower than both `-ncmoe 56` and global disable-opt |
| `-ncmoe 52` | load failure | SYCL3 allocation `18002258944` bytes failed |

The important detail is that `-ncmoe 54` is not primarily a model-buffer fit issue. It loads, then Q8 reorder asks for a 20,054,016-byte temporary buffer and Level Zero returns OOM. `GGML_SYCL_DISABLE_OPT=1` proves skipping reorder avoids that crash, but speed does not improve.

I added a targeted patch so reorder host fallback can be disabled at runtime:

```text
patches/llama-cpp-sycl-reorder-host-fallback-env-20260507.patch
```

With `GGML_SYCL_REORDER_HOST_FALLBACK=0`, failed reorder temp allocations now return `false` and the caller skips reorder for that tensor instead of falling through to host USM and aborting. This is cleaner than global `GGML_SYCL_DISABLE_OPT=1`, but it did not improve MiniMax throughput.

## Failed Row-Split Workaround

I tried row split with all attention and KV kept on CPU, while keeping expert blocks 0-11 on GPU and blocks 12-61 CPU-mapped:

```bash
-ot 'blk\.\d+\.(attn_norm|attn_q_norm|attn_k_norm|attn_(q|k|v|output))\.weight=CPU;blk\.(1[2-9]|[2-5][0-9]|6[01])\.ffn_(gate|up|down)_exps\.weight=CPU' \
-sm row -ts 1/1/1/1 -nkvo 1 -fa 0
```

It avoided the Q8 attention device-loss and moved through CPU attention, but then segfaulted at the first SYCL split containing block-0 FFN/expert work:

```text
split_begin id=7 backend=0/SYCL0 n_inputs=1 n_nodes=38 first=ffn_inp-0/ADD last=norm-1/RMS_NORM
timeout: the monitored command dumped core
```

Log:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-cpuattn-cpukv-experts0-11gpu-p0n1-20260507T122041Z.log
```

This keeps row split on the todo list, but confirms it needs a code fix in the SYCL split scheduler or split `MUL_MAT_ID` path.

## Failed Round-Robin Tail Expert Placement

I tried to work around `-ncmoe 50` packing all remaining experts into SYCL3 by overriding tail expert blocks round-robin across the four B70s:

```bash
-ot 'blk\.(50|54|58)\.ffn_(gate|up|down)_exps\.weight=SYCL0;blk\.(51|55|59)\.ffn_(gate|up|down)_exps\.weight=SYCL1;blk\.(52|56|60)\.ffn_(gate|up|down)_exps\.weight=SYCL2;blk\.(53|57|61)\.ffn_(gate|up|down)_exps\.weight=SYCL3'
```

It still failed during allocation:

```text
alloc_tensor_range: failed to allocate SYCL2 buffer of size 5839339520
```

Log:

```text
/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-layer-ncmoe50-tail12-rr-p0n1-20260507T122828Z.log
```

This reinforces that the current Level Zero/SYCL multi-device allocation behavior is fragile even for mid-sized per-device model buffers.

## Next Code Work

1. Add a targeted opt-out for Q8 reorder host-memory fallback, instead of globally using `GGML_SYCL_DISABLE_OPT=1`. If the reorder temp allocation fails, skip reorder cleanly for that tensor and continue with the plain Q8 path.
2. Continue investigating row-split MiniMax expert execution. The CPU-attention/CPU-KV run gets past the Q8 attention blocker and then crashes around the first GPU FFN/expert split.
3. Study `ik_llama.cpp` graph split and active-expert-only offload design. The relevant ideas are graph scheduling, limiting GPUs per layer, and reducing inter-GPU exchange type.
4. Avoid downloading a full non-GGUF MiniMax variant until disk is cleaned or expanded. Current free space is about 75 GB, which is not enough for the official 220 GB weight path and not enough margin for most alternative M2.7 quants.
