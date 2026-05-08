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

Note: `cmosmudwl0004k004hzz6l4u6` was the then-current best quality-preserving Q4_0 GGUF result. It fuses `linear_attn_out -> RESHAPE -> ADD` into the existing SYCL allreduce+residual-add path, reducing decode graph plain 20 KiB allreduces from 49 to 1 without changing weights, quantization, KV dtype, speculative decoding, sampling, or GPU power.

Note: `cmot9sgsi000lib042rqd6c62` was the then-current best quality-preserving Q4_0 GGUF result. It was captured after reboot recovered a transient xe/Level Zero degraded state. The submitted `tokSTotal` is conservative/slightly underreported; computed from prompt/decode rates it is about `68.259 tok/s`. Output throughput is the headline metric and was submitted correctly.

Note: `cmota1fpx0001l404wepbjtb7` is a clean negative-scaling diagnostic. The 4x run used the same quality-preserving Q4_0 path but was slower than 3x, confirming that the fourth B70 currently adds more 20 KiB allreduce latency than it removes in row-parallel matvec work.

Date: 2026-05-04

Model: `vrfai/Qwen3.6-27B-FP8`, static compressed-tensors FP8.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-qwen36-vrfai-fp8-tp4-fa2-kvfp8-negative-p512-n256` | `cmornlh8g000vkw04yb57ukvl` | 4 | 512 | 256 | 28.036 | 84.108 |
| `vllm-qwen36-vrfai-fp8-tp4-fa2-ngram4-lookup2-4-p512-n512` | `cmos3pnqo000kkz04o4aiup22` | 4 | 512 | 512 | 47.675 | 95.350 |

Note: diagnostic negative result. `--kv-cache-dtype fp8` increased reported KV capacity but was slower than auto/BF16 KV and vLLM warns it may reduce accuracy without proper scaling.

Note: `cmos3pnqo000kkz04o4aiup22` is the current validated FP8 best. It uses auto/BF16 KV, XPU FlashAttention2, n-gram speculative decoding with `num_speculative_tokens=4`, lookup min/max `2/4`, and `CCL_ATL_TRANSPORT=ofi` with default IPC/topology recognition.

Date: 2026-05-06 and 2026-05-07

Verified by `GET /api/benchmarks?username=steveseguin&dateFrom=2026-05-06T00:00:00Z&limit=100` on 2026-05-06 and by direct POST response on 2026-05-07.

Model: `unsloth/Qwen3.6-27B`, local GGUF file `Qwen3.6-27B-Q4_0.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-syncafter2-reducewait-p512-n512` | `cmotnyi25001jqu01fccla8cf` | 3 | 512 | 512 | 46.194 | 66.400 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-syncafter2-equal-negative-p512-n512` | `cmotpxzii000pqy019v46swqn` | 4 | 512 | 512 | 34.929 | 49.748 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-assist005-sg2-p512-n512` | `cmou581wv002dld0197mffpco` | 4 | 512 | 512 | 39.204 | 52.816 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-swiglu-ub128-p512-n512` | `cmougm58m00dpld012rbm9rbs` | 3 | 512 | 512 | 46.805 | 75.218 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp2-rmsnormmul-ub128-p512-n512` | `cmouju3dx00f3ld01rzmp9u76` | 2 | 512 | 512 | 42.106 | 75.571 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-rmsnormmul-ub128-p512-n512` | `cmoujcois00esld01c5s6bwht` | 3 | 512 | 512 | 49.366 | 79.667 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-getrows-rmsmul-ub128-p512-n512` | `cmoultsa900h0ld011f0r2hcs` | 3 | 512 | 512 | 49.404 | 79.017 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-guardfix-ub128-p512-n512` | `cmous57ci00lqld01a8x5azdq` | 3 | 512 | 512 | 49.553 | 78.947 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-fuseadd-root-residual-p512-n512` | `cmouvurhh00nqld010dtr4xrl` | 3 | 512 | 512 | 50.809 | 80.502 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-rootresid-poll25-p512-n512` | `cmouxjqao000npn01hxqn68td` | 3 | 512 | 512 | 50.922 | 81.243 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp3-fused-ba-no-root-p512-n512` | `cmov6p4r7007tqr01yi8ug4un` | 3 | 512 | 512 | 50.130 | 80.205 |
| `llamacpp-qwen36-27b-q4_0-sycl-tp4-assist005-guardfix-p512-n512` | `cmoute8kg00mbld017ye0dfbz` | 4 | 512 | 512 | 44.088 | 64.174 |

Note: `cmotnyi25001jqu01fccla8cf` is the pre-fused-SwiGLU Q4_0 GGUF baseline. It uses the same Q4_0 weights, f16 KV cache, flash attention, no speculative decoding, no power-limit changes, and software-only SYCL/Level Zero patches including Q8 activation cache, fused MMVQ2, event-barrier allreduce, fused allreduce+ADD, and `GGML_SYCL_COMM_SYNC_AFTER=2`.

Note: `cmotpxzii000pqy019v46swqn` is a useful negative equal-split four-card result. It confirms that the fourth B70 is not automatically beneficial for single-session Q4_0 because row shards become too narrow and add launch/quantization/collective overhead.

Note: `cmou581wv002dld0197mffpco` is the best current four-card Q4_0 result. It uses a small fourth-card assist split (`-ts 1/1/1/0.05`) plus `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2`, improving equal 4x from `34.929313 tok/s` to `39.204149 tok/s`, but it still trails the three-card result.

Note: `cmougm58m00dpld012rbm9rbs` is the pre-RMS_NORM+MUL Q4_0 GGUF result. It adds the opt-in `GGML_SYCL_FUSE_MMVQ2_SWIGLU=1` path, which fuses Q4_0 FFN gate/up matvecs with split SwiGLU. A short greedy decode matched baseline stdout byte-for-byte. The valid 512/512 TP3 run requires `-ub 128`; default `-ub 512` currently fails meta compute-buffer reservation after the active patch stack.

Note: `cmouju3dx00f3ld01rzmp9u76` is the best submitted two-card Q4_0 GGUF result. It uses the same RMS_NORM+MUL patch stack as the TP3 result, devices `SYCL2/SYCL1`, tensor split `1/1`, and `-ub 128`. A detailed payload returned HTTP 500, but the reduced core-metric payload was accepted.

Note: `cmoujcois00esld01c5s6bwht` is the pre-GET_ROWS best submitted quality-preserving Q4_0 GGUF result. It adds the opt-in `GGML_SYCL_FUSE_RMS_NORM_MUL=1` path on top of fused MMVQ2 and fused MMVQ2+SwiGLU. Greedy decode matched baseline stdout byte-for-byte. The full annotated payload returned HTTP 500 from the API, but a reduced payload with core metrics and notes was accepted.

Note: `cmoultsa900h0ld011f0r2hcs` was the first submitted GET_ROWS fused quality-preserving Q4_0 GGUF result. It enables the opt-in `GGML_META_FUSE_ALLREDUCE_GET_ROWS=1` path on top of the RMS_NORM+MUL stack. Five-repeat same-build A/B: GET_ROWS on `49.403656 tok/s`, GET_ROWS off `48.827917 tok/s`. Greedy `llama-completion` output matched byte-for-byte. This changes scheduling only: same Q4_0 weights, f16 KV, no speculative decoding, no sampling change, and no power-limit change.

Note: `cmous57ci00lqld01a8x5azdq` is the 2026-05-07 guard-fix refresh of the same quality-preserving TP3 Q4_0 stack. A misplaced Q8-cache guard temporarily disabled the validated `allreduce+ADD` path and dropped a decode-only control to `27.676519 tok/s`; removing that over-broad guard restored `backend+add` paths and produced `49.552666 tok/s` on `p512/n512/r3`. The detailed annotated LocalMaxxing payload returned HTTP 500, while the reduced core-metric payload was accepted.

Note: `cmouvurhh00nqld010dtr4xrl` is the first submitted root-residual Q4_0 GGUF result. It adds `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1` to the guard-fix stack, reusing the mirrored root residual in fused allreduce+ADD instead of reading peer residual buffers. This was submitted before the stronger token/logit harness existed. A later probe found that `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1` plus `GGML_META_FUSE_ALLREDUCE_ADD=1` can diverge, so this record is now historical/suspect rather than currently quality-cleared.

Note: `cmouxjqao000npn01hxqn68td` is the fastest submitted Q4_0 GGUF performance ceiling, but not the current quality-cleared result. It keeps the same TP3 root-residual stack as `cmouvurhh00nqld010dtr4xrl` and changes only `--poll 50` to `--poll 25`. A short poll sweep favored `25`, and the full validation produced `50.922114 tok/s` output and `81.243035 tok/s` total. The detailed annotated LocalMaxxing payload returned HTTP 500, while the reduced core-metric payload was accepted.

Note: `cmov6p4r7007tqr01yi8ug4un` is the current quality-cleared no-root Q4_0 GGUF result. It uses an experimental augmented GGUF with added flat Qwen35 `ssm_ba` tensors generated from the same Q4_0 beta/alpha weights, disables root-residual, and keeps the rest of the validated TP3 stack. The final token/logit probe matched the original model byte-for-byte. A detailed payload returned HTTP 500, while the reduced core-metric payload was accepted.

Note: `cmoute8kg00mbld017ye0dfbz` is the current best submitted four-card Q4_0 result. It reruns the assist split `1/1/1/0.05` after the guard fix with Q8 activation cache, fused MMVQ2, fused MMVQ2+SwiGLU, fused RMS_NORM+MUL, fused allreduce+ADD, fused final GET_ROWS, and single-kernel allreduce. It improves the older assist result by `12.46%` and equal four-card by `26.22%`, but still trails the current TP3 result by `11.03%`. The detailed annotated LocalMaxxing payload returned HTTP 500, while the reduced core-metric payload was accepted.

Date: 2026-05-06

Model: `vrfai/Qwen3.6-27B-FP8`, static compressed-tensors FP8.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-qwen36-vrfai-fp8-tp4-ngram-cpu-p512-n512` | `cmotql1v60013qy01016jcs7r` | 4 | 512 | 512 | 49.582 | 99.164 |
| `vllm-qwen36-vrfai-fp8-tp4-ngram-longcontext-p2048-n512` | `cmottw16x002wqy01jvbluobl` | 4 | 2048 | 512 | 43.688 | 218.442 |
| `vllm-qwen36-vrfai-fp8-tp4-nospec-32k-p2048-n256` | `cmoudx2qr00c3ld01xxq8hiu0` | 4 | 2048 | 256 | 42.996 | 386.966 |

Note: `cmotql1v60013qy01016jcs7r` is the current best submitted static FP8 result. It uses vLLM TP4/PP1 with CPU n-gram speculative decoding. The target model remains FP8 and speculation is verified, so this does not change final-output model quality in the way an INT4 target would.

Note: `cmottw16x002wqy01jvbluobl` is the long-context FP8 validation. It is slower than the 512/512 best but shows the static FP8 TP4 path works at a 2048-token prompt window with high total throughput.

Note: `cmoudx2qr00c3ld01xxq8hiu0` is the clean 32k-context FP8 validation. It uses no speculative decoding and records XPU/Level Zero in `engineFlags.extraFlags` because the LocalMaxxing backend enum currently rejects `xpu`.

Date: 2026-05-07

Model: `vrfai/Qwen3.6-27B-FP8`, static compressed-tensors FP8.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-qwen36-vrfai-fp8-pp2tp2-capacity-negative-p512-n512` | `cmout3vhy00m6ld01162ujv21` | 4 | 512 | 512 | 27.722 | 55.445 |

Note: `cmout3vhy00m6ld01162ujv21` is a capacity-focused negative result for the 2x2 FP8 layout. PP2xTP2 fits `max_model_len=32768` and reports large KV-cache headroom, but it is much slower than TP4/PP1 for batch-1 single-session speed. It uses the same FP8 target weights, auto/BF16 KV, no speculative decoding, and no power-limit changes.

Date: 2026-05-07

Model: `MiniMaxAI/MiniMax-M2.7`, local Unsloth GGUF `MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llamacpp-minimax-m27-ud-iq4_xs-sycl-layer-ncmoe56-p0-n64` | `cmovgojv80008n501lupo67x6` | 4 | 0 | 64 | 0.472 | 0.472 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-devmapfix-p0-n64-r3` | `cmowf7tgs000do301f1zd6jbr` | 4 | 0 | 64 | 14.292 | 14.292 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-kqv-offload-p0-n64-r3` | `cmowft2hr000oo3019is4snoq` | 4 | 0 | 64 | 16.384 | 16.384 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fused-mul-unary-p0-n64-r3` | `cmowqyak0008co201oxuuzaid` | 4 | 0 | 64 | 16.405 | 16.405 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fast-mmid-p0-n64-r3` | `cmowt5ciy00d0o201f1mcrg3q` | 4 | 0 | 64 | 17.336 | 17.336 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fast-mmid-mmv-y2-p0-n64-r5` | `cmowx1t6z000mml01v111mzvl` | 4 | 0 | 64 | 17.547 | 17.547 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fast-mmid-mmv-y2-p512-n128-r3` | `cmowyq5tu001jml01b470i75g` | 4 | 512 | 128 | 17.516 | 36.854 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fusedrms-mmv-y2-ub64-p0-n64-r5` | `cmox103ol0040ml019yzs6gvs` | 4 | 0 | 64 | 17.698 | 17.698 |
| `llamacpp-minimax-m27-ud-iq4_xs-rpc-layer-fusedrms-mmv-y2-ub64-p512-n128-r3` | `cmox1gcxl0049ml01kiijqbpo` | 4 | 512 | 128 | 17.693 | 38.489 |
| `vllm-minimax-m27-autoround-int4-xpu-tp4-p512-n128` | `cmox4zohw0077ml01czu880bz` | 4 | 512 | 128 | 13.450 | 67.259 |
| `vllm-minimax-m27-autoround-int4-xpu-tp4-p512-n128-pidfd` | `cmox6tys30085ml0125gihg18` | 4 | 512 | 128 | 19.850 | 99.231 |
| `vllm-minimax-m27-autoround-int4-xpu-tp4-p512-n128-hybrid-moe` | `cmox94fsm0095ml01tjeb20rr` | 4 | 512 | 128 | 20.110 | 100.538 |

Note: This is a diagnostic baseline, not an optimized all-GPU result. It uses llama.cpp layer split with `-ncmoe 56`, which CPU-maps the first 56 of 62 MoE expert layers and leaves only the final 6 expert layers GPU-resident on SYCL3. It was submitted because it is the first reproducible MiniMax completion on the 4x B70 system and records the current gap. LocalMaxxing's command parser misread `-t 8` as sampler temperature; the run is `llama-bench`, so sampling temperature is not meaningful here.

Note: `cmowf7tgs000do301f1zd6jbr` is the corrected post device-map-fix all-GPU layer result. It uses the process-per-GPU RPC+SYCL path with `-ts 1/1/1/1`, fused MoE, merged up/gate experts, fused MMAD, F16 KV, and no power-limit changes. The first retry after LocalMaxxing recovered returned HTTP 400 because the API now requires another metric alongside output tok/s; adding `tokSTotal=14.292387` was accepted.

Note: `cmowft2hr000oo3019is4snoq` keeps the same corrected layer split and quality-preserving settings, but enables K/Q/V offload with `-nkvo 0`. KV remains F16, and the r3 samples were `16.2585`, `16.4391`, `16.4532` tok/s.

Note: `cmowqyak0008co201oxuuzaid` adds quality-preserving SYCL RPC worker support for `GGML_OP_FUSED_MUL_UNARY` while keeping fused RMSNorm disabled because it was neutral/slower. Same-build A/B was `16.404929 tok/s` with fused mul unary enabled versus `16.374820 tok/s` with it disabled, so this is a small current high rather than a major optimization.

Note: `cmowt5ciy00d0o201f1mcrg3q` enables the default-off `GGML_SYCL_FAST_MUL_MAT_ID_IQ4_XS=1` path for MiniMax expert-down `MUL_MAT_ID`. It improves the prior MiniMax high from `16.404929` to `17.335655 tok/s`. A synthetic IQ4_XS `MUL_MAT_ID` probe produced identical SYCL checksums and first outputs with fast path on versus off; a manual dequantized oracle showed the SYCL path is close (`nmse=1.44e-05`) while the CPU graph path diverges in this synthetic case.

Note: `cmowx1t6z000mml01v111mzvl` adds `GGML_SYCL_MMV_Y_RUNTIME=2` runtime row packing for the SYCL MMVQ-style kernels on top of the fast-MMID MiniMax path. Same-build r5 control was `17.198973 tok/s`; runtime MMV Y=2 was `17.547020 tok/s`, with samples `17.3265`, `17.6006`, `17.6046`, `17.6047`, `17.5987`. This is a software-only scheduling/kernel change with the same UD-IQ4_XS weights, F16 KV, layer split, and no power-limit changes. A deterministic 16-token greedy generation smoke matched default row grouping byte-for-byte. LocalMaxxing rejected `backend=sycl-rpc`, so the record was submitted without a backend and the SYCL/RPC details are in notes and engine flags.

Note: `cmowyq5tu001jml01b470i75g` is the same MiniMax fast-MMID + MMV Y=2 stack at a 512-token prompt and 128-token decode window. Prompt throughput was `50.905433 tok/s`; decode throughput was `17.515510 tok/s`, so the decode bottleneck is essentially unchanged at this context length while prefill is faster. `tokSTotal=36.854313` was computed from prompt and decode timings.

Note: `cmox103ol0040ml019yzs6gvs` is the current best MiniMax GGUF result. It keeps fast IQ4_XS `MUL_MAT_ID`, `GGML_SYCL_MMV_Y_RUNTIME=2`, DNN disabled, and `-ub 64`, but stops forcing `GGML_DISABLE_FUSED_RMS_NORM=1`. The r5 samples were `17.5369`, `17.7524`, `17.7386`, `17.7282`, `17.7328`. A deterministic 16-token greedy generation smoke matched the prior fused-RMS-disabled Y=2 baseline byte-for-byte, so this is currently marked quality-preserving for the benchmark scope.

Note: `cmox1gcxl0049ml01kiijqbpo` is the same current best GGUF stack at `p512/n128/r3`. Prompt throughput was `54.506141 tok/s`; decode throughput was `17.693021 tok/s`; total throughput was `38.489462 tok/s`. It supersedes the earlier context datapoint `cmowyq5tu001jml01b470i75g`.

Note: `cmox4zohw0077ml01czu880bz` is a diagnostic vLLM/XPU TP4 result using `Lasimeri/MiniMax-M2.7-int4-AutoRound` but submitted under base model `MiniMaxAI/MiniMax-M2.7` because LocalMaxxing's HF lookup rejected the quant repo. Unpatched vLLM/INC XPU fell back to unquantized `FusedMoE` and OOMed; the local experimental INC patch routes AutoRound `FusedMoE` through `MoeWNA16Config`, allowing the model to load at about 28.11 GiB/card and generate. It is slower than the current GGUF result, and the log identifies the next blocker: missing B70 tuned MoE config for `E=256,N=384,dtype=int4_w4a16`.

Note: `cmox6tys30085ml0125gihg18` was the first strong vLLM/XPU MiniMax AutoRound result. It uses the same local INC `FusedMoE` WNA16 patch as `cmox4zohw0077ml01czu880bz`, but switches `CCL_ZE_IPC_EXCHANGE` from sockets to `pidfd`. At p512/n128 it reached `19.85` output tok/s and `99.231127` total tok/s, ahead of the current MiniMax GGUF p512 decode result. It was later superseded by the hybrid B70 MoE config result `cmox94fsm0095ml01tjeb20rr`.

Note: `cmox94fsm0095ml01tjeb20rr` is the current best MiniMax AutoRound vLLM/XPU result. It adds a hybrid B70 MoE config in `configs/vllm/minimax-m27-b70-int4-w4a16-moe-hybrid-20260508.json`: tuned key `1` for decode and default prompt-size keys `64`, `256`, and `512`. The accepted LocalMaxxing row omits `backend=xpu` because the API currently rejects that backend enum; XPU/Level Zero details are retained in `engineFlags.extraFlags` and in the archived payload/response files.

Not submitted: `GGML_SYCL_MOE_UP_GATE_PAIR_DOT=1` paired up/gate dot loop for MiniMax `MOE_FUSED_UP_GATE` produced `16.840924 tok/s` with high variance (`15.8979`, `17.3159`, `17.3090`). This was neutral/slower than `cmowt5ciy00d0o201f1mcrg3q`, so it remains a negative/noise experiment rather than a public benchmark result.
