# llama.cpp SYCL Q4 current patch after vec4 revert

Patch snapshot generated on 2026-05-06 from:

`/home/steve/src/llama.cpp-q4-b70`

Base commit:

`db44417b027cff147f7de85e7da22bc6a3a804fb`

Local files:

- `llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch`
- `llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz`
- `llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz.b64`

GitHub upload note:

The `.patch.gz.b64` artifact is split into `part00` through `part05` to avoid large single-file API payloads. Reconstruct with:

```bash
cat llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz.b64.part* > llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz.b64
base64 -d llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz.b64 > llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz
gunzip -k llama-cpp-sycl-q4-current-after-vec4-revert-20260506.patch.gz
```

Important: this patch intentionally does not contain the temporary `GGML_SYCL_COMM_VEC4_F32` experiment. That gate regressed performance and was removed before this snapshot.
