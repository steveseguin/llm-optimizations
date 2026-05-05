# llama.cpp SYCL Allreduce-To-Reshape Experimental Patch

File:

- `llama-cpp-sycl-allreduce-to-reshape-experimental-20260505.patch`
- GitHub upload form: `llama-cpp-sycl-allreduce-to-reshape-experimental-20260505.patch.gz.b64`

Checksum:

- `sha256:7ce7451634d1a536957168f7219213325bdf1729e92549c0c5231d1f42920692`
- `patch.gz.b64 sha256:3b4ce48e0e6849271fbcb391a713a23475bb179bd483b0540d70c343af11dea8`

Decode:

```bash
base64 -d llama-cpp-sycl-allreduce-to-reshape-experimental-20260505.patch.gz.b64 | gunzip > llama-cpp-sycl-allreduce-to-reshape-experimental-20260505.patch
```

Scope:

- `ggml/include/ggml-backend.h`
- `ggml/src/ggml-backend-meta.cpp`
- `ggml/src/ggml-sycl/ggml-sycl.cpp`

Important:

- this patch was generated from the active Q4 B70 llama.cpp working tree;
- it is best treated as applying on top of the previously recorded fused-add and SYCL collective patches, not as a clean upstream patch;
- behavior is gated by `GGML_META_FUSE_ALLREDUCE_RESHAPE=1`.

Outcome:

- graph recognition and backend dispatch work;
- 4x 512/128 improved only marginally from `33.497463` to `33.743952 tok/s`;
- 3x selector `2,1,3` 512/512 validation regressed to `43.734996 tok/s` versus the existing `44.004344 tok/s` fused-add-only validation;
- do not promote this patch as a performance win without further collective redesign.
