# llama.cpp SYCL Local-Write / Root-Residual Diagnostic Patch

Date: 2026-05-05

The full diagnostic diff is stored as `llama-cpp-sycl-localwrite-rootresid-diagnostics-20260505.patch.gz.b64` to keep the GitHub connector payload small.

Reconstruct the patch with:

```bash
base64 -d patches/llama-cpp-sycl-localwrite-rootresid-diagnostics-20260505.patch.gz.b64 \
  | gunzip > patches/llama-cpp-sycl-localwrite-rootresid-diagnostics-20260505.patch
```

Expected patch metadata after reconstruction:

```text
sha256: e500f8a2c69aacf87822cebaacc5a734128eab02f243342108c89b42ccf9233a
lines: 1849
bytes: 86577
source: git -C /home/steve/src/llama.cpp-q4-b70 diff -- ggml/src/ggml-sycl/ggml-sycl.cpp
```

This is a diagnostic patch, not a recommended performance patch. The local-write and root-residual switches were neutral or negative versus the fused allreduce+ADD baseline and should stay off for best runs.
