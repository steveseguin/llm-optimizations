# 2026-05-08 MiniMax Fast IQ4_XS `mul_mat_id`

## Summary

Enabled and tested the default-off `GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1` path for MiniMax M2.7 UD-IQ4_XS. This targets the remaining `MUL_MAT_ID` bucket from the MoE expert-down path, which op timing showed was still a meaningful decode cost after merged up/gate and fused MoE work.

## Result

New current MiniMax GGUF high:

```text
17.335655 tok/s, p0/n64/r3, 4x B70 RPC+SYCL layer split, F16 KV
LocalMaxxing: cmowt5ciy00d0o201f1mcrg3q
```

Comparison:

| Variant | tok/s | samples |
| --- | ---: | --- |
| fast `MUL_MAT_ID` IQ4_XS on | 17.335655 | 17.1807, 17.3988, 17.4275 |
| fused mul unary previous high | 16.404929 | 16.2231, 16.5070, 16.4847 |

This is a `5.67%` gain over the prior MiniMax high, without changing model weights, quantization, KV dtype, sampler behavior, expert routing, or GPU power limits.

## Follow-up: Paired Up/Gate Dot Attempt

Tried an opt-in `GGML_SYCL_MOE_UP_GATE_PAIR_DOT=1` experiment that computes the MiniMax IQ4_XS MoE up and gate dot products in one loop inside `MOE_FUSED_UP_GATE`.

Result:

```text
16.840924 tok/s, p0/n64/r3, samples 15.8979, 17.3159, 17.3090
```

This is neutral to slightly worse than the `17.335655 tok/s` fast-`MUL_MAT_ID` baseline. The later samples were close to baseline, but the average was lower and variance was higher, so this was not submitted to LocalMaxxing as an improvement.

Interpretation: simply pairing the gate/up dot loops does not reduce the remaining `MOE_FUSED_UP_GATE` bucket enough to overcome added register pressure and instruction scheduling cost on B70. Keep this path default-off unless a later rewrite proves a clear win.

## Validation

Added `prototypes/minimax_mmid_iq4xs_check.cpp`, a small GGML probe that builds a synthetic IQ4_XS `ggml_mul_mat_id` graph and compares CPU and SYCL against a manual dequantized reference built from `ggml_internal_get_type_traits(GGML_TYPE_IQ4_XS).to_float`.

The important A/B result is that fast path off and fast path on produced identical SYCL-side checksums and first outputs:

```text
sycl_sum=-63.626713506877422
sycl_sumsq=220952.15214172262
first=-4.7533946,-1.53130412,-4.87711143,1.29914367,-6.31177616,-9.72053337,1.87014472,-1.95123363
```

The manual-reference comparison showed the SYCL result is close to the dequantized oracle with fast path off and on:

```text
sycl_ref_max_abs=0.0794775 sycl_ref_nmse=1.44346e-05
```

The original CPU-vs-SYCL mismatch came from the CPU backend graph path for this synthetic `MUL_MAT_ID` case, not from the fast SYCL path:

```text
cpu_ref_max_abs=4.50142 cpu_ref_nmse=0.0288577
sycl_cpu_max_abs=4.48206 sycl_cpu_nmse=0.031076
```

Interpretation: the fast path preserves the existing SYCL result and the SYCL result matches a manual dequantized oracle closely enough for this synthetic case. The remaining CPU graph mismatch should be tracked separately, but it no longer blocks using the fast path for this workload.

## Repro

Workers:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:<id> \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
ZES_ENABLE_SYSMAN=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1 \
rpc-server --device SYCL0 --host 127.0.0.1 --port <50100+id>
```

Client:

```bash
GGML_DISABLE_FUSED_RMS_NORM=1 \
llama-bench \
  -m MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -rpc '127.0.0.1:50100|0,127.0.0.1:50101|0,127.0.0.1:50102|0,127.0.0.1:50103|0' \
  -p 0 -n 64 -r 3 -ngl 99 -sm layer -ts 1/1/1/1 \
  -fa 0 -nkvo 0 -ub 32 -ctk f16 -ctv f16 -t 4 -w 0 -mmp 1 \
  -rtr 1 -fmoe 1 -no-mmad 0 -muge 1 -sas 0 -o json
```

Probe:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -std=c++17 -O2 prototypes/minimax_mmid_iq4xs_check.cpp \
  -I/home/steve/src/ik_llama.cpp/ggml/include \
  -I/home/steve/src/ik_llama.cpp/ggml/src \
  -L/home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/ggml/src -lggml \
  -Wl,-rpath,/home/steve/src/ik_llama.cpp/build-sycl-rpc-b70/ggml/src \
  -o /tmp/minimax_mmid_iq4xs_check

ONEAPI_DEVICE_SELECTOR=level_zero:0 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
/tmp/minimax_mmid_iq4xs_check

ONEAPI_DEVICE_SELECTOR=level_zero:0 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_DISABLE_FUSED_RMS_NORM=1 \
GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1 \
/tmp/minimax_mmid_iq4xs_check
```

## Next

- Investigate why CPU backend `MUL_MAT_ID` diverges from the manual IQ4_XS oracle in this synthetic case.
- Consider promoting the fast path from hidden env var to an opt-in runtime flag once MiniMax generation-level validation is stable.
- Treat `GGML_SYCL_MOE_UP_GATE_PAIR_DOT=1` as a negative/noise experiment for now.
- Continue the vLLM/XPU AutoRound INT4 path once the USB-hosted download finishes; it remains the most likely route to a larger MiniMax step-change.
