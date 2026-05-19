# MiniMax M2.7 Q/K RMS Helper on Tiny-FP32 In-Place Allreduce

Date: 2026-05-19

## Result

`VLLM_MINIMAX_QK_RMS_XPU_HELPER=1` was retested on top of the current alias-correct tiny-FP32 in-place allreduce path:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2`
- `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`
- `VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

Four benchmark repeats:

- Run 1: `88.794746` output tok/s, `118.392995` total tok/s
- Run 2: `88.771642` output tok/s, `118.362189` total tok/s
- Run 3: `87.411381` output tok/s, `116.548508` total tok/s
- Run 4: `88.274652` output tok/s, `117.699535` total tok/s
- Mean: `88.313105` output tok/s, `117.750807` total tok/s

This is `+0.24%` output tok/s versus the alias-correct tiny-FP32 in-place baseline of `88.103866` output tok/s, and `-0.49%` versus the faster tiny-FP32 skip-clone headline of `88.748424` output tok/s.

## Quality

The full strict quality gate passed before benchmarking:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.

## Log Scan

Alias warning patterns were absent from the quality and benchmark logs:

- `output of this custom operator`
- `may not alias`
- `vllm::all_reduce`

Two quality logs printed `Bad address (src/pipe.cpp:367)` after successful completion and JSON write. This remains shutdown noise to monitor; it did not affect the quality JSONs or benchmark JSONs.

## Interpretation

The Q/K helper is quality-safe and gives a small clean-path gain over the alias-correct in-place baseline. The gain is too small to call a new speed headline, but it is useful because it preserves the no-alias-warning path and confirms that the local Q/K RMS math can be changed safely when the tiny FP32 variance allreduce stays exact.

The result also reinforces the current bottleneck model: replacing only local Q/K RMS math is not enough to reach the next major tier. The next useful work is fusing or relocating the decode-critical boundaries around Q/K variance allreduce, residual allreduce, and MoE epilogues.

## Command

```bash
LABEL=minimax-qk-helper-tinyfp32-inplace-20260519 \
ATTENTION_BACKEND=default \
MAX_BATCHED_TOKENS=512 \
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1 \
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4 \
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0 \
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2 \
VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0 \
VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0 \
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
CCL_TOPO_P2P_ACCESS=1 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=4 \
QUALITY_TIMEOUT=30m \
BENCH_TIMEOUT=25m \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-helper-tinyfp32-inplace-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T042622Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-helper-tinyfp32-inplace-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T042622Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T044210Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T044459Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T044749Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T045045Z.json`
- Local data: `data/minimax-m27-qk-helper-tinyfp32-inplace-20260519.json`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-qk-helper-tinyfp32-inplace-p512n1536-20260519.payload.json`
- LocalMaxxing response: `data/localmaxxing-responses/minimax-m27-autoround-qk-helper-tinyfp32-inplace-p512n1536-20260519.response.json`

## Decision

Keep this as the current clean-path incremental candidate. It does not replace the `88.748424` skip-clone speed headline, but it improves the alias-correct baseline while preserving strict quality.

- LocalMaxxing id: `cmpc5xmm6005jpc01k84dxd14`
