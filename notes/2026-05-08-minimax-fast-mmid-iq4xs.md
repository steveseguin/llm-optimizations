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

## Validation

Added `prototypes/minimax_mmid_iq4xs_check.cpp`, a small GGML probe that builds a synthetic IQ4_XS `ggml_mul_mat_id` graph and compares CPU against SYCL.

The important A/B result is that fast path off and fast path on produced identical SYCL-side checksums and first outputs:

```text
sycl_sum=-63.626713506877422
sycl_sumsq=220952.15214172262
first=-4.7533946,-1.53130412,-4.87711143,1.29914367,-6.31177616,-9.72053337,1.87014472,-1.95123363
```

The probe returns nonzero because CPU and SYCL disagree for this synthetic IQ4_XS `MUL_MAT_ID` case even when the fast path is disabled:

```text
max_abs=4.48206 max_rel=822.229 nmse=0.031076
```

Interpretation: the fast path appears to preserve the existing SYCL behavior. The CPU-vs-SYCL mismatch is a separate pre-existing oracle issue and should be investigated before treating this path as upstream-ready.

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

- Investigate the CPU-vs-SYCL IQ4_XS `MUL_MAT_ID` mismatch with a simpler dequantized oracle.
- Keep the fast path default-off until the oracle question is resolved.
- Continue the vLLM/XPU AutoRound INT4 path once the USB-hosted download finishes; it remains the most likely route to a larger MiniMax step-change.
