# llama.cpp SYCL Q4 current guard-fix patch

Generated on 2026-05-07 from `/home/steve/src/llama.cpp-q4-b70` after restoring Q8-cache compatibility for the validated SYCL `allreduce+ADD` path.

The important corrective change is in `ggml/src/ggml-sycl/ggml-sycl.cpp`: `ggml_backend_sycl_comm_allreduce_add_tensor()` must not reject `GGML_SYCL_Q8_CACHE=1`. The Q8-cache guard remains appropriate for the experimental lower-level `comm_mul_mat_allreduce_add` diagnostic path.

Apply from a compatible llama.cpp checkout with:

```bash
base64 -d patches/llama-cpp-sycl-q4-current-guardfix-20260507.patch.gz.b64 | gunzip | git apply
```

Validation artifacts:

- `/home/steve/bench-results/qwen36-q4_0-gguf/regression-debug-20260507/tp3-fixed-meta-stats-p0n1-20260507T005433Z.log`
- `/home/steve/bench-results/qwen36-q4_0-gguf/regression-debug-20260507/tp3-fixed-control-p0n256-20260507T005551Z.jsonl`
- `/home/steve/bench-results/qwen36-q4_0-gguf/regression-debug-20260507/tp3-fixed-full-p512n512-r3-20260507T005722Z.jsonl`
