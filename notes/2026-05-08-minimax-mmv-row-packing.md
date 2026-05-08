# 2026-05-08 MiniMax MMV Row Packing

## Summary

Added a runtime SYCL row-packing knob for MiniMax IQ4_XS matvec-heavy decode:

```text
GGML_SYCL_MMV_Y_RUNTIME=2
```

This keeps the same MiniMax M2.7 UD-IQ4_XS GGUF weights, F16 KV cache, sampler path, layer split, and GPU power settings. The change groups two output rows per workgroup for the relevant SYCL MMVQ-style kernels, reducing launch/workgroup overhead and improving cache use for the 4x B70 RPC layer-split path.

## Result

New current MiniMax GGUF high:

```text
17.547020 tok/s, p0/n64/r5, 4x B70 RPC+SYCL layer split, F16 KV
LocalMaxxing: cmowx1t6z000mml01v111mzvl
```

Comparison:

| Variant | tok/s | samples |
| --- | ---: | --- |
| fast MMID + runtime `GGML_SYCL_MMV_Y_RUNTIME=2` | 17.547020 | 17.3265, 17.6006, 17.6046, 17.6047, 17.5987 |
| same-build control, default MMV row grouping | 17.198973 | 16.4378, 17.2950, 17.4133, 17.4222, 17.4266 |
| previous LocalMaxxing high, fast MMID only | 17.335655 | 17.1807, 17.3988, 17.4275 |

This is a `2.02%` gain versus the same-build r5 control and a `1.22%` gain versus the prior submitted MiniMax high. It is not the step-change needed to reach the 30 tok/s target, but it is a repeatable software-only improvement.

Context check:

```text
p512/n128/r3: prompt 50.905433 tok/s, decode 17.515510 tok/s, total 36.854313 tok/s
LocalMaxxing: cmowyq5tu001jml01b470i75g
```

The decode rate is effectively the same as p0/n64. This points back to decode-side matvec/MoE scheduling rather than prompt setup as the limiting factor for this GGUF path.

## Follow-up Row-Pack Sweep

Additional row-packing variants did not beat generic Y=2:

| Variant | tok/s | samples | interpretation |
| --- | ---: | --- | --- |
| `GGML_SYCL_MMV_Y_RUNTIME=8` | 17.238444 | 16.6394, 17.5203, 17.5557 | first repeat regressed, later repeats only tied the Y=2 cluster |
| `GGML_SYCL_MMV_Y_RUNTIME=2`, `GGML_SYCL_MOE_IQ4_XS_MMV_Y=4` | 17.232041 | 16.6258, 17.5346, 17.5357 | MoE-specific Y=4 did not improve over generic Y=2 |

Conclusion: Y=2 is the current B70 setting for this path. More row packing appears to increase variance and does not improve steady decode.

## Microbatch Sweep

With Y=2 held fixed, `-ub 64` is a tiny local win over `-ub 32`, while `-ub 128` is neutral/slower:

| Variant | tok/s | samples | status |
| --- | ---: | --- | --- |
| `-ub 64`, r5 | 17.559741 | 17.3921, 17.6070, 17.6027, 17.5936, 17.6033 | local best, not submitted because delta is tiny |
| `-ub 64`, r3 | 17.560269 | 17.4253, 17.6356, 17.6199 | confirms direction |
| `-ub 128`, r3 | 17.502587 | 17.3805, 17.6000, 17.5272 | not better |

Use `-ub 64` for the next GGUF MiniMax sweeps, but keep the public LocalMaxxing p0/n64 record at `17.547020 tok/s` until a larger improvement clears the noise floor.

## Timing Delta

Short op-timing runs show the direction of the win:

| Op bucket | fast MMID baseline | MMV Y=2 | delta |
| --- | ---: | ---: | ---: |
| `MUL_MAT` | 115.366 ms | 111.580 ms | -3.28% |
| `MOE_FUSED_UP_GATE` | 95.922 ms | 88.382 ms | -7.86% |
| `MUL_MAT_ID` | 19.653 ms | 18.798 ms | -4.35% |

The largest remaining decode buckets are still dense matvec and fused MoE up/gate. Row packing helps both, but does not address the broader layer-mode scheduling and all-GPU graph-shape limits.

## Correctness Smoke

Ran a deterministic 16-token MiniMax generation through the RPC client with the default row grouping and with `GGML_SYCL_MMV_Y_RUNTIME=2`. Both used the same prompt, seed, F16 KV, layer split, fast-MMID setting, and greedy sampler. The generated stdout matched byte-for-byte:

```text
78e517594d13b4c3405739fda666369c2cfe6afaa111e1e75a0b5eb0a848d6fd  default.stdout
78e517594d13b4c3405739fda666369c2cfe6afaa111e1e75a0b5eb0a848d6fd  y2.stdout
```

This is not a full-logit equivalence proof, but it is enough to mark the row-packing result as quality-preserving for the current benchmark scope. The only issue found was tool-side: `llama-cli --rpc` segfaulted when given the `llama-bench` style `host:port|device` string. Plain comma-separated RPC servers worked.

## Repro

Workers:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<id> \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
ZES_ENABLE_SYSMAN=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1 \
GGML_SYCL_MMV_Y_RUNTIME=2 \
rpc-server --device SYCL0 --host 127.0.0.1 --port <50100+id>
```

Client:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
llama-bench \
  -m MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 5 -ngl 99 -sm layer -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

## Files

- Main result: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv-runtime2-r5-p0n64-20260508T124605Z.jsonl`
- Same-build control: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-control-r5-p0n64-20260508T120407Z.jsonl`
- Compile-time MMV2 confirmation: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv2-r5-p0n64-20260508T115654Z.jsonl`
- Runtime MMV8 negative: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv-runtime8-r3-p0n64-20260508T131435Z.jsonl`
- Runtime MMV2 + MoE4 negative: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv2-moe4-r3-p0n64-20260508T132138Z.jsonl`
- Context run: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv-runtime2-r3-p512n128-20260508T133014Z.jsonl`
- Microbatch `-ub 64` local best: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv-runtime2-ub64-r5-p0n64-20260508T135521Z.jsonl`
- Microbatch `-ub 128` neutral: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-fast-mmid-mmv-runtime2-ub128-r3-p0n64-20260508T134833Z.jsonl`
- Correctness smoke: `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/correctness/mmv-y2-smoke-20260508T125856Z`
- LocalMaxxing payload: `/home/steve/bench-results/localmaxxing-minimax-m27-fast-mmid-mmv-y2-20260508.payload.json`
- LocalMaxxing response: `/home/steve/bench-results/localmaxxing-minimax-m27-fast-mmid-mmv-y2-20260508.response.json`

## Next

- Keep `GGML_SYCL_MMV_Y_RUNTIME=2` as the default MiniMax GGUF test setting for now. A deterministic generation smoke matched byte-for-byte against default row grouping.
- Keep `GGML_SYCL_MMV_Y_RUNTIME=4` and `8` marked neutral/negative for now. Compile-time MMV4 produced `17.191979 tok/s`; runtime MMV8 produced `17.238444 tok/s`.
- Keep MoE-specific `GGML_SYCL_MOE_IQ4_XS_MMV_Y=4` marked negative; it produced `17.232041 tok/s` with generic MMV Y=2.
- Continue toward vLLM/XPU AutoRound INT4 TP4, since GGUF row packing is now delivering single-digit-percent gains rather than the larger improvement needed for the 30 tok/s goal.
