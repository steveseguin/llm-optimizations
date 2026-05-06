# LocalMaxxing Submissions

Date: 2026-05-03

Model: `Lorbus/Qwen3.6-27B-int4-AutoRound`

All submitted results returned `APPROVED`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-int4-single-b70-mtp-500-256` | `cmoq41b9d001alg043wsnthz2` | 1 | 500 | 256 | 45.2 | 133.44 |
| `vllm-int4-single-b70-mtp-500-512` | `cmoq47sll0005l104v3i0f9l3` | 1 | 500 | 512 | 41.3 | 81.60 |
| `vllm-int4-tp2-b70-nonmtp-500-256` | `cmoq4e9dw0002js04ledqyycn` | 2 | 500 | 256 | 49.1 | 144.88 |
| `vllm-int4-tp2-b70-nonmtp-500-512` | `cmoq4krfb000cl40456wobg7e` | 2 | 500 | 512 | 48.3 | 95.56 |
| `vllm-int4-single-b70-nonmtp-500-256` | `cmoq4r8rc0001l804tocgibus` | 1 | 500 | 256 | 31.8 | 93.80 |
| `vllm-int4-tp2-b70-mtp-500-256` | `cmoq4xppt0003ky04xidngli9` | 2 | 500 | 256 | 35.6 | 105.03 |

Date: 2026-05-03

Model: `z-lab/Qwen3.6-27B-DFlash` submitted as the accepted Hugging Face base for local `Qwen3.6-27B-Q4_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-async-cpy-512` | `cmoqkcqpv0006la04l5mtlj2q` | 2 | 0 | 512 | 37.690 | 37.690 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-split-anchor-512` | `cmoqli4dm0005l404kdf9ofnd` | 3 | 0 | 512 | 38.365 | 38.365 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-comm-allreduce-ub32-512` | `cmoqnkcx10006kw04f2jmahrv` | 2 | 0 | 512 | 38.621 | 38.621 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-single-kernel-allreduce-512` | `cmoqp6jpq0004lb04241n9ns3` | 2 | 0 | 512 | 39.849 | 39.849 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-single-kernel-allreduce-512` | `cmoqptj6i000blb04j0i2u9yo` | 3 | 0 | 512 | 41.367 | 41.367 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-single-kernel-root213-512` | `cmoqqed6s0007jv049wnizwle` | 3 | 0 | 512 | 41.737 | 41.737 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-single-kernel-allreduce-512` | `cmor2e5r00004jl04o99d26p8` | 4 | 0 | 512 | 31.482 | 31.482 |

Date: 2026-05-04

Model: `Qwen/Qwen3.6-27B`, local GGUF file `ggml-org/Qwen3.6-27B-GGUF` `Qwen3.6-27B-Q8_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q8_0-sycl-tp2-single-kernel-allreduce-p512-n128` | `cmor8w11d000lji04rn2zwh32` | 2 | 512 | 128 | 25.733 | 87.259 |

Note: the full annotated payload with detailed `engineFlags` returned HTTP 500 from the API, but a reduced payload with the core metrics was accepted and approved. The full local payload remains in `/home/steve/localmaxxing_payloads.json`.

Date: 2026-05-04

Model: `Qwen/Qwen3.6-27B-FP8`, official FP8 Safetensors, vLLM/XPU local experimental block-FP8 requant path.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-qwen36-27b-fp8-xpu-tp2-requant-p512-n512` | `cmorb75xb001ckz0489eqc9se` | 2 | 512 | 512 | 20.106 | 40.215 |

Note: tok/s was computed from end-to-end benchmark latency, so it includes prefill overhead. This is a conservative public datapoint for the current vLLM/XPU FP8 path, not a steady-state decode win.

Date: 2026-05-04

Model: `Qwen/Qwen3.6-27B`, local GGUF file `Qwen3.6-27B-Q4_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-q8cache-root213-p512-n256-min` | `cmordq9t5000dl404x309pj48` | 3 | 512 | 256 | 42.432 | 78.320 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-q8cache-validate-p512-n512-min` | `cmormylxz000fib04wodwo1ng` | 2 | 512 | 512 | 40.487 | 64.506 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-q8cache-validate-p512-n512-min` | `cmorn71e2000kib0415vo51vj` | 3 | 512 | 512 | 41.659 | 63.800 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-q8cache-negative-p512-n128-min` | `cmornec37000okw040zl9563z` | 4 | 512 | 128 | 31.913 | 70.610 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-eventbarrier-validate-p512-n512-min` | `cmortp5vn000el404dj3zqv0u` | 3 | 512 | 512 | 43.605 | 66.000 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-postguc-fuseadd-p512-n512-min` | `cmoslhw0i0008jj04h59bb96n` | 3 | 512 | 512 | 44.181 | 66.660 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-k617023-guc70494-fuseadd-p512-n512-min` | `cmosm05ke0005ib048aljq6pl` | 3 | 512 | 512 | 44.238 | 66.735 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-reshapeadd-p512-n512-min` | `cmosmudwl0004k004hzz6l4u6` | 3 | 512 | 512 | 44.813 | 67.389 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-reshapeadd-postreboot-p512-n512-min` | `cmot9sgsi000lib042rqd6c62` | 3 | 512 | 512 | 45.624 | 67.925 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-reshapeadd-negative-p512-n512-min` | `cmota1fpx0001l404wepbjtb7` | 4 | 512 | 512 | 34.376 | 51.448 |

Note: full detailed payloads hit an API HTTP 500, so reduced payloads were accepted. Full command and engine flags remain in `/home/steve/localmaxxing_payloads.json` under the matching labels.

Note: `cmosmudwl0004k004hzz6l4u6` is the current best quality-preserving Q4_0 GGUF result. It fuses `linear_attn_out -> RESHAPE -> ADD` into the existing SYCL allreduce+residual-add path, reducing decode graph plain 20 KiB allreduces from 49 to 1 without changing weights, quantization, KV dtype, speculative decoding, sampling, or GPU power.

Note: `cmot9sgsi000lib042rqd6c62` is the current best quality-preserving Q4_0 GGUF result. It was captured after reboot recovered a transient xe/Level Zero degraded state. The submitted `tokSTotal` is conservative/slightly underreported; computed from prompt/decode rates it is about `68.259 tok/s`. Output throughput is the headline metric and was submitted correctly.

Note: `cmota1fpx0001l404wepbjtb7` is a clean negative-scaling diagnostic. The 4x run used the same quality-preserving Q4_0 path but was slower than 3x, confirming that the fourth B70 currently adds more 20 KiB allreduce latency than it removes in row-parallel matvec work.

Date: 2026-05-04

Model: `vrfai/Qwen3.6-27B-FP8`, static compressed-tensors FP8.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-qwen36-vrfai-fp8-tp4-fa2-kvfp8-negative-p512-n256` | `cmornlh8g000vkw04yb57ukvl` | 4 | 512 | 256 | 28.036 | 84.108 |
| `vllm-qwen36-vrfai-fp8-tp4-fa2-ngram4-lookup2-4-p512-n512` | `cmos3pnqo000kkz04o4aiup22` | 4 | 512 | 512 | 47.675 | 95.350 |

Note: diagnostic negative result. `--kv-cache-dtype fp8` increased reported KV capacity but was slower than auto/BF16 KV and vLLM warns it may reduce accuracy without proper scaling.

Note: `cmos3pnqo000kkz04o4aiup22` is the current validated FP8 best. It uses auto/BF16 KV, XPU FlashAttention2, n-gram speculative decoding with `num_speculative_tokens=4`, lookup min/max `2/4`, and `CCL_ATL_TRANSPORT=ofi` with default IPC/topology recognition.

Date: 2026-05-06

Model: `unsloth/Qwen3.6-27B`, local GGUF file `Qwen3.6-27B-Q4_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-assist005-sg2-p512-n512` | `cmou581wv002dld0197mffpco` | 4 | 512 | 512 | 39.204 | 52.816 |

Note: quality-preserving four-card Q4_0 result. It uses the same GGUF weights, f16 KV cache, flash attention, no speculative decoding, and no GPU power changes. The useful change is treating the fourth B70 as a small assist device with `-ts 1/1/1/0.05` plus `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2`; this improves the previous equal-split four-card validation from `34.929313 tok/s` to `39.204149 tok/s`, but still trails the best three-card result.
