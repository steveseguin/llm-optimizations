# MiniMax M2.7 Clone-Safe Custom Allreduce Win

Date: 2026-05-18

## Result

`VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` was retested with an alias-safe input clone before the vLLM allreduce custom op:

- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

The previous direct custom-op attempt failed raw145 n64 exact quality with a PyTorch aliasing warning. Cloning the input tensor before entering `torch.ops.vllm.all_reduce` fixes the alias contract and keeps the graph/custom-op path quality-safe.

Four clean confirmation runs:

- Run 1: `87.691616` output tok/s, `116.922154` total tok/s
- Run 2: `87.117954` output tok/s, `116.157272` total tok/s
- Run 3: `87.262412` output tok/s, `116.349883` total tok/s
- Run 4: `87.044535` output tok/s, `116.059380` total tok/s
- Mean: `87.279129` output tok/s, `116.372172` total tok/s

This is `+5.92%` output tok/s over the previous strict no-attention-delay promoted result of `82.404268` output tok/s and `+8.28%` over the earlier MoE-WS FlashAttention/PIECEWISE baseline of `80.602755` output tok/s.

## Quality

The full strict quality gate passed before benchmarking, and the four-repeat confirmation passed it again:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.

## Command

```bash
LABEL=minimax-compile-allreduce-custom-op-clone-confirm4-ar-20260518 \
ATTENTION_BACKEND=default \
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=4 \
QUALITY_TIMEOUT=45m \
BENCH_TIMEOUT=35m \
QUALITY_STARTUP_GUARD_SECONDS=900 \
SHM_STALL_MAX_WARNINGS=6 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-clone-confirm4-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T221935Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-clone-confirm4-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T221935Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T223525Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T223818Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T224110Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T224402Z.json`
- Patch: `patches/minimax-clone-safe-custom-allreduce-20260518.patch`

## Decision

Promote as the current strict MiniMax baseline. This is a quality-preserving gain and is repeatable across four clean long runs.

- LocalMaxxing id: `cmpbsqm4l001qpc0199azisgz`

The next speed path should use this clone-safe custom-op result as the new baseline, then focus on true XPU fused-boundary work: Q/K RMS variance allreduce plus apply, hidden allreduce plus residual/RMSNorm, and attention output allreduce plus post-attention normalization.
