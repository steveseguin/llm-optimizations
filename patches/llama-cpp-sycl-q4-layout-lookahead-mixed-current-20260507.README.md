# llama.cpp SYCL Q4 Layout/Lookahead/Mixed Patch

File:

- `llama-cpp-sycl-q4-layout-lookahead-mixed-current-20260507.patch.gz.b64`

Decode/apply:

```bash
base64 -d llama-cpp-sycl-q4-layout-lookahead-mixed-current-20260507.patch.gz.b64 | gunzip > /tmp/q4-layout-lookahead-mixed.patch
git -C /path/to/llama.cpp apply /tmp/q4-layout-lookahead-mixed.patch
```

This is a current-tree diff from `/home/steve/src/llama.cpp-q4-b70` for the relevant SYCL files, not a minimal upstream-ready patch. It includes cumulative gated Q4 scheduler/kernel experiments in:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `ggml/src/ggml-sycl/mmvq.cpp`
- `ggml/src/ggml-sycl/mmvq.hpp`
- `ggml/src/ggml-sycl/vecdotq.hpp`

Relevant gates from this pass:

- `GGML_SYCL_REORDER_Q4_0_VDR4=1`: clear regression; keep off.
- `GGML_SYCL_FUSE_MMVQ2_LOOKAHEAD=1`: neutral; keep off.
- `GGML_SYCL_FUSE_MMVQ2_MIXED=1`: removes 48 `attn_qkv + attn_gate` launches with deferred copies, but end-to-end decode is flat; keep off.

See:

- `notes/2026-05-07-q4-layout-lookahead-mixed-negative.md`
- `data/qwen36-q4-layout-lookahead-mixed-20260507.json`
