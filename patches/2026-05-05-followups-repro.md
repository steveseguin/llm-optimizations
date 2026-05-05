# 2026-05-05 Follow-up Patch Reproduction

Use this with the existing 2026-05-04 / early 2026-05-05 experimental patches in this repo.

## llama.cpp SYCL

- Focused remote/apply patch: `patches/llama-cpp-sycl-followups-focused-20260505.patch`
- Full local cumulative diff: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-followups-minimax-hostbounce-q4-skiprootready-20260505.patch`
- Full cumulative diff SHA256: `d7c66fa7ce43f0d674ff7c7bd4f7fd8c2a5cde2d281e3c2b9f50a30a7f7dfe80`

The focused patch adds:

- `GGML_SYCL_MUL_MAT_ID_SPLIT_HOST_BOUNCE=1` for the MiniMax split `MUL_MAT_ID` prototype.
- `GGML_SYCL_COMM_SKIP_ROOT_READY=1` for the Q4_0 single-kernel allreduce diagnostic.

Observed result: host-bounce did not make MiniMax decode viable, and skip-root-ready was neutral on 4x B70.

## vLLM XPU

- Remote/apply patch: `patches/vllm-xpu-qwen36-fp8-fa2-ngram-language-only-20260505.patch`
- Local patch SHA256: `6d41e5582c34d2ed63ff8835a93a14ecb262787312c378f407dfcb37903147de`

The vLLM patch preserves the known static FP8 TP4 path:

- singleton attention scale reshape for XPU FA2;
- language-only Qwen3.5/Qwen3.6 loading;
- block-FP8 XPU fallback/requant kernels;
- n-gram speculative Gated DeltaNet metadata fixes.
