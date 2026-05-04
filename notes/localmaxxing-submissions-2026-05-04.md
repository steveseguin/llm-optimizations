# LocalMaxxing Submissions

Date: 2026-05-03 through 2026-05-04

This audit records submitted B70 benchmark results for reproducibility and cross-checking. Exact payloads, including command snippets and engine flags, are archived as `data/localmaxxing-payloads-20260504.json.gz.b64`; decode with:

```bash
base64 -d data/localmaxxing-payloads-20260504.json.gz.b64 | gunzip > /tmp/localmaxxing-payloads-20260504.json
```

## INT4 AutoRound

Model: `Lorbus/Qwen3.6-27B-int4-AutoRound`

All submitted results returned `APPROVED`. These are useful speed datapoints, but they are not quality-equivalent Q4_0 GGUF results.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-int4-single-b70-mtp-500-256` | `cmoq41b9d001alg043wsnthz2` | 1 | 500 | 256 | 45.2 | 133.44 |
| `vllm-int4-single-b70-mtp-500-512` | `cmoq47sll0005l104v3i0f9l3` | 1 | 500 | 512 | 41.3 | 81.60 |
| `vllm-int4-tp2-b70-nonmtp-500-256` | `cmoq4e9dw0002js04ledqyycn` | 2 | 500 | 256 | 49.1 | 144.88 |
| `vllm-int4-tp2-b70-nonmtp-500-512` | `cmoq4krfb000cl40456wobg7e` | 2 | 500 | 512 | 48.3 | 95.56 |
| `vllm-int4-single-b70-nonmtp-500-256` | `cmoq4r8rc0001l804tocgibus` | 1 | 500 | 256 | 31.8 | 93.80 |
| `vllm-int4-tp2-b70-mtp-500-256` | `cmoq4xppt0003ky04xidngli9` | 2 | 500 | 256 | 35.6 | 105.03 |

## Q4_0 GGUF

Model: `Qwen/Qwen3.6-27B`, local GGUF file `Qwen3.6-27B-Q4_0.gguf`. Earlier submissions used `z-lab/Qwen3.6-27B-DFlash` as the accepted Hugging Face base for the same local Q4_0 GGUF until `Qwen/Qwen3.6-27B` was accepted.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-async-cpy-512` | `cmoqkcqpv0006la04l5mtlj2q` | 2 | 0 | 512 | 37.690 | 37.690 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-split-anchor-512` | `cmoqli4dm0005l404kdf9ofnd` | 3 | 0 | 512 | 38.365 | 38.365 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-comm-allreduce-ub32-512` | `cmoqnkcx10006kw04f2jmahrv` | 2 | 0 | 512 | 38.621 | 38.621 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-single-kernel-allreduce-512` | `cmoqp6jpq0004lb04241n9ns3` | 2 | 0 | 512 | 39.849 | 39.849 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-single-kernel-allreduce-512` | `cmoqptj6i000blb04j0i2u9yo` | 3 | 0 | 512 | 41.367 | 41.367 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-single-kernel-root213-512` | `cmoqqed6s0007jv049wnizwle` | 3 | 0 | 512 | 41.737 | 41.737 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-single-kernel-allreduce-512` | `cmor2e5r00004jl04o99d26p8` | 4 | 0 | 512 | 31.482 | 31.482 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-q8cache-root213-p512-n256-min` | `cmordq9t5000dl404x309pj48` | 3 | 512 | 256 | 42.432 | 78.320 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-q8cache-validate-p512-n512-min` | `cmormylxz000fib04wodwo1ng` | 2 | 512 | 512 | 40.487 | 64.506 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-q8cache-validate-p512-n512-min` | `cmorn71e2000kib0415vo51vj` | 3 | 512 | 512 | 41.659 | 63.800 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-q8cache-negative-p512-n128-min` | `cmornec37000okw040zl9563z` | 4 | 512 | 128 | 31.913 | 70.610 |

The over-42 tok/s Q4_0 result is a 3-GPU result, not a 4-GPU result. The 4-GPU Q4_0 submissions are preserved as negative-scaling diagnostics.

## Q8_0 GGUF

Model: `Qwen/Qwen3.6-27B`, local GGUF file `ggml-org/Qwen3.6-27B-GGUF` `Qwen3.6-27B-Q8_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q8_0-sycl-tp2-single-kernel-allreduce-p512-n128` | `cmor8w11d000lji04rn2zwh32` | 2 | 512 | 128 | 25.733 | 87.259 |

## FP8

| Model | Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.6-27B-FP8` | `vllm-qwen36-27b-fp8-xpu-tp2-requant-p512-n512` | `cmorb75xb001ckz0489eqc9se` | 2 | 512 | 512 | 20.106 | 40.215 |
| `vrfai/Qwen3.6-27B-FP8` | `vllm-qwen36-vrfai-fp8-tp4-fa2-kvfp8-negative-p512-n256` | `cmornlh8g000vkw04yb57ukvl` | 4 | 512 | 256 | 28.036 | 84.108 |
| `vrfai/Qwen3.6-27B-FP8` | `vllm-qwen36-vrfai-fp8-tp4-fa2-p512-n512` | `cmork3n3k000ujo04y73lbr1j` | 4 | 512 | 512 | 41.503 | 69.172 |
| `vrfai/Qwen3.6-27B-FP8` | `vllm-qwen36-vrfai-fp8-pp2-tp2-fa2-p512-n256` | `cmormmlz0000bky04wpu4oc01` | 4 | 512 | 256 | 22.721 | 68.164 |

The FP8 KV-cache run is diagnostic-only: it increases reported KV capacity but was slower than auto/BF16 KV and vLLM warns it can reduce accuracy without proper scaling.

## API Notes

Detailed llama.cpp payloads containing `engineFlags` returned HTTP 500 from the API during this audit. Reduced payloads with core metrics were accepted, while the exact command snippets and engine flags were preserved in the payload archive above.
