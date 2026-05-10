# llm-scaler `fix_27b_kernel` Branch

Date: 2026-05-10

Intel's `llm-scaler` repo now has branch `origin/fix_27b_kernel` at commit `db05b45`, titled `esimd: fix resadd_norm_gemv_int4 race on large N`.

The patch targets `vllm/custom-esimd-kernels-vllm/csrc/xpu/esimd_kernels/resadd_norm_gemv_int4.h`, not the MiniMax unsigned-u4 MoE bridge. Its commit message says the fused ResAddNormGEMV INT4 kernel can race when `N` is large because one workgroup writes the updated residual while other workgroups later read the same residual for their sum pass. The stated observed failure is Qwen3.6-27B dense `gate_up` with `N=8704, K=5120, TP=4`, while smaller `N=256` MoE-router cases stay below the threshold.

The fix splits large-`N` cases into:

- a prepass that computes residual update plus RMS-normalized hidden state;
- a GEMV-from-normed kernel that reads the already-normalized vector.

Relevance to our work:

- This is a correctness fix for dense Qwen3.6 INT4 / sym-int4 paths, especially if we revisit AutoRound-style dense INT4 for Qwen3.6 27B.
- It is not expected to change MiniMax M2.7 AutoRound vLLM throughput because our current MiniMax speed path uses the local `moe_int4.sycl` raw-u4 decode bridge and does not call this dense ResAddNormGEMV kernel.
- It does not apply to Qwen3.6 Q4_0 GGUF or static FP8 quality-preserving results.

Decision: record it as a candidate patch to apply and test only when returning to Qwen3.6 dense INT4. Do not mix it into the dirty llm-scaler MiniMax branch unless a Qwen INT4 experiment needs it.
