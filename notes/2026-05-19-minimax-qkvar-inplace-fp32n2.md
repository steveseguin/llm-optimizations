# MiniMax M2.7 Q/K Variance Tiny-FP32 In-Place Allreduce

Date: 2026-05-19

## Result

`VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2` replaces the previous fast-but-warned tiny-FP32 clone-elision path with a mutating no-return custom op:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

The intent is narrow: keep the clone-safe custom allreduce path for normal tensors, but use a PyTorch-schema-correct mutating operation for tiny FP32 allreduces with `numel <= 2`. In this model path that targets the Q/K RMS variance scalar collective while leaving larger residual/hidden collectives on the safer cloned path.

Four benchmark repeats:

- Run 1: `87.565326` output tok/s, `116.753768` total tok/s
- Run 2: `88.951949` output tok/s, `118.602599` total tok/s
- Run 3: `87.342699` output tok/s, `116.456932` total tok/s
- Run 4: `88.555489` output tok/s, `118.073985` total tok/s
- Mean: `88.103866` output tok/s, `117.471821` total tok/s

This is `-0.73%` output tok/s versus the previous tiny-FP32 skip-clone promoted result of `88.748424` output tok/s, but `+0.94%` over the clone-safe custom-allreduce baseline of `87.279129` output tok/s. The important win is that the PyTorch custom-op alias warning is gone.

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

Three of four benchmark logs printed `Bad address (src/pipe.cpp:367)` after successful request completion and JSON write. This is recorded as shutdown noise to monitor; it did not affect the benchmark JSONs or quality checks.

## Command

```bash
LABEL=minimax-qkvar-inplace-fp32n2-ar-20260519 \
ATTENTION_BACKEND=default \
MAX_BATCHED_TOKENS=512 \
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0 \
VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2 \
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

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qkvar-inplace-fp32n2-ar-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T021834Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qkvar-inplace-fp32n2-ar-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T021834Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T023426Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T023722Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T024013Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T024306Z.json`
- Runtime patch: `patches/minimax-qkvar-inplace-fp32n2-20260519.patch`
- LocalMaxxing payload: `data/localmaxxing-minimax-m27-autoround-qkvar-inplace-fp32n2-p512n1536-20260519.payload.json`

## Decision

Do not replace the previous speed headline yet; it remains slightly faster. Keep this as the cleaner reliability version of the same optimization family because it removes the PyTorch alias warning while staying in the same performance band and preserving quality.

- LocalMaxxing id: `cmpc1dxgv0052pc01s1j9i37l`
