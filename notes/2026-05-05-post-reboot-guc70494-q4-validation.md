# 2026-05-05 Post-Reboot GuC 70.49.4 Qwen Q4_0 Validation

## Context

After the MiniMax q8_0 matvec experiments caused Level Zero device-lost failures, the system was rebooted with B70 display probing disabled:

```text
options xe disable_display=1 probe_display=0
```

The BMG GuC firmware was also updated from Ubuntu's older packaged blob to upstream `70.49.4`:

```text
/lib/firmware/xe/bmg_guc_70.bin
sha256: 328d57b5af4b373c02db5b8e42e25f67ce3dc7afc7e1c1882940d0ea70ebd6d8
```

Post-reboot checks:

- all four B70s enumerate through Level Zero;
- `sycl-peer-read-test` passes across all four GPUs;
- `journalctl -b -k` confirms `xe/bmg_guc_70.bin version 70.49.4` on all four B70s.

## Validation

The previously validated fast Qwen3.6 27B Q4_0 GGUF command remains reproducible after reboot and the GuC update.

Command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 \
  -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 \
  -p 512 -n 512 -r 3 --poll 50 -o jsonl
```

Result:

- prompt: `135.705541 tok/s`, stddev `0.128581`;
- decode: `44.180797 tok/s`, stddev `0.035825`;
- decode samples: `44.2132`, `44.1869`, `44.1423`;
- total: `66.659637 tok/s`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-guc70494-nondnn-build-triple213-exactfast-p512n512-r3-20260505T121540Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-guc70494-nondnn-build-triple213-exactfast-p512n512-r3-20260505T121540Z.log`.

LocalMaxxing accepted the reduced payload:

```text
cmoslhw0i0008jj04h59bb96n
```

## DNN Build Finding

A fresh current-source `GGML_SYCL_DNN=ON`, `GGML_SYCL_DEVICE_ARCH=intel_gpu_bmg_g31` AOT build was created at:

```text
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-aot-dnn-current
```

It survives the tiny multi-card Qwen prompt smoke that previously isolated the oneMKL GEMM device-lost path, but it is not a performance candidate:

- 2x selector `2,1`, prompt/decode smoke completed;
- 3x selector `2,1,3`, prompt/decode smoke completed;
- tiny no-warmup results were much slower than the non-DNN fast path.

Interpretation: oneDNN is useful as a stability fallback or debug comparison for prompt-side GEMM, but the validated speed path remains the non-DNN SYCL build with the Q8 cache, async copy, single-kernel allreduce, event barrier, and fused allreduce+ADD.

## Decision

Keep the Q4_0 3x B70 command as the current quality-preserving llama.cpp/SYCL reference. The reboot, headless `xe` options, and GuC 70.49.4 did not regress the validated `44 tok/s` class result.
