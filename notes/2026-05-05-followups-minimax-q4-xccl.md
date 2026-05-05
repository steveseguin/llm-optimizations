# 2026-05-05 Follow-ups: MiniMax, Q4 Sync, XCCL Gate

## Runtime Baseline

- OS: Ubuntu 24.04.4 LTS, kernel `6.17.0-22-generic`.
- GPU runtime packages already match the current Intel compute-runtime GitHub release line checked on 2026-05-05:
  - `intel-opencl-icd 26.14.37833.4-0`
  - `libze-intel-gpu1 26.14.37833.4-0`
  - `intel-ocloc 26.14.37833.4-0`
  - `intel-igc-core-2 2.32.7`
  - `intel-igc-opencl-2 2.32.7`
  - `libigdgmm12 22.9.0`
- oneAPI generation: 2026.0 installed locally.

## MiniMax M2.7 Split `MUL_MAT_ID` Follow-up

Added follow-up knobs to the experimental split `MUL_MAT_ID` path:

```bash
GGML_SYCL_MUL_MAT_ID_SPLIT_DEBUG=1
GGML_SYCL_MUL_MAT_ID_SPLIT_HOST_BOUNCE=1
```

The host-bounce path routes cross-device activation/result copies through the existing SYCL host staging fallback instead of issuing direct cross-device USM copies from the expert helper. This was intended to test whether the previous first-token stall came from Level Zero cross-device copy waits.

Test:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZES_ENABLE_SYSMAN=1 \
UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT_DEBUG=1 \
GGML_SYCL_MUL_MAT_ID_SPLIT_HOST_BOUNCE=1 \
LLAMA_EXPERT_PLACEMENT_DEBUG=1 \
llama-bench -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 -sm row -ts 1/1/1/1 \
  -p 0 -n 1 -r 1 -ngl 99 -ncmoe 60 -fa 1 -ub 32 -ctk f16 -ctv f16 --no-warmup -t 8 --poll 50 -o jsonl
```

Outcome:

- Timed out after 360 seconds.
- No JSONL output and no log output were emitted before timeout.
- Kernel log recorded new `xe` engine resets during the attempt:
  - `2026-05-04 21:29:32`: `0000:03:00.0` GT0 CCS engine reset.
  - `2026-05-04 21:34:32`: `0000:83:00.0` GT0 BCS engine reset.
- A tiny Qwen Q4_0 single-card smoke after the reset still completed, so the system was not left totally unusable.

Conclusion: the host-bounce copy change did not make MiniMax decode viable. The next MiniMax work should avoid full-model first-token attempts until the helper has smaller synthetic coverage or per-stage heartbeat logging that flushes before model load/decode timeouts.

## Qwen3.6 27B Q4_0 Sync Follow-up

Added an env-gated single-kernel allreduce synchronization experiment:

```bash
GGML_SYCL_COMM_SKIP_ROOT_READY=1
```

Rationale: SYCL queues are in-order. In the single-kernel allreduce path, the root stream does not need a separate `ext_oneapi_submit_barrier()` before launching the reduction kernel; only non-root streams need readiness events. This removes one submitted event per allreduce without changing math.

4x B70 corrected DNN-off control, same build:

- `GGML_SYCL_COMM_SKIP_ROOT_READY=0`
- prompt: `101.476558 tok/s`
- decode: `32.720377 tok/s`
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-controlpost-quad0123-dnn0-p512n128-20260505T014610Z.jsonl`

4x B70 skip-root-ready:

- `GGML_SYCL_COMM_SKIP_ROOT_READY=1`
- prompt: `101.475153 tok/s`
- decode: `32.776781 tok/s`
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-skiprootready-quad0123-dnn0-p512n128-20260505T014426Z.jsonl`

Conclusion: neutral. The measured decode difference is about 0.17%, inside normal noise. Do not submit to LocalMaxxing and do not treat this as a meaningful 4x optimization.

## FP8 / XCCL Gate

Standalone XCCL is still unhealthy:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
CCL_ZE_IPC_EXCHANGE=sockets \
python -m torch.distributed.run --standalone --nproc_per_node=2 /home/steve/b70_xccl_allreduce_bench.py
```

Outcome:

- exit status `1`
- both ranks segfaulted with signal 11 before any measured allreduce row was emitted.
- log: `/home/steve/bench-results/qwen36-fp8-vllm/xccl-standalone-2rank-post-q4-minimax-20260505T014812Z.log`

Conclusion: keep vLLM FP8 tensor-parallel validation paused until a reboot or driver reload restores standalone XCCL allreduce.

## LocalMaxxing

No new submission from this batch:

- MiniMax produced no valid decode metric.
- Q4 skip-root-ready was neutral versus same-build control.
- XCCL smoke is a blocker, not a model benchmark.

## Reproduction Artifacts

- llama.cpp/SYCL cumulative experimental diff:
  - `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-followups-minimax-hostbounce-q4-skiprootready-20260505.patch`
  - `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-followups-minimax-hostbounce-q4-skiprootready-20260505.patch.b64`
- vLLM/XPU FP8/n-gram/language-only cumulative diff:
  - `/home/steve/llm-optimization-artifacts/patches/vllm-xpu-qwen36-fp8-fa2-ngram-language-only-20260505.patch`
  - `/home/steve/llm-optimization-artifacts/patches/vllm-xpu-qwen36-fp8-fa2-ngram-language-only-20260505.patch.b64`
