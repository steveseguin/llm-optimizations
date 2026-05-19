# MiniMax M2.7 Q/K Variance Tiny-FP32 Clone Elision Win

Date: 2026-05-18

## Result

`VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2` was tested on top of the current clone-safe compiled allreduce custom-op baseline:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

The intent is narrow: keep the clone-safe custom allreduce path for normal tensors, but skip the clone for tiny FP32 allreduces with `numel <= 2`. In this model path that targets the Q/K RMS variance scalar collective while leaving larger residual/hidden collectives on the safer cloned path.

Four benchmark repeats:

- Run 1: `88.148932` output tok/s, `117.531909` total tok/s
- Run 2: `89.072947` output tok/s, `118.763930` total tok/s
- Run 3: `89.473994` output tok/s, `119.298659` total tok/s
- Run 4: `88.297823` output tok/s, `117.730431` total tok/s
- Mean: `88.748424` output tok/s, `118.331232` total tok/s

This is `+1.53%` output tok/s over the previous clone-safe custom-allreduce promoted result of `87.279129` output tok/s.

## Quality

The full strict quality gate passed before benchmarking:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.

## Caveat

PyTorch warns that the no-clone `vllm::all_reduce` path may return an output that aliases an input. This candidate is quality-clean and repeatable on the current stack, but PyTorch says this warning can become an error in a future release. The next cleaner version should make the custom op produce a non-aliasing output for the tiny FP32 path, or fuse Q/K variance allreduce with the RMS apply path so the boundary disappears entirely.

## Command

```bash
LABEL=minimax-qkvar-skipclone-fp32n2-ar-20260518 \
ATTENTION_BACKEND=default \
MAX_BATCHED_TOKENS=512 \
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1 \
VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=0 \
VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=2 \
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
BENCH_REPEATS=2 \
QUALITY_TIMEOUT=45m \
BENCH_TIMEOUT=35m \
QUALITY_STARTUP_GUARD_SECONDS=900 \
SHM_STALL_MAX_WARNINGS=6 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-qkvar-skipclone-fp32n2-ar-20260518 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

Two additional benchmark-only confirmation repeats were run with the same environment and cache root to produce the four-run mean.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qkvar-skipclone-fp32n2-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260519T011834Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qkvar-skipclone-fp32n2-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260519T011834Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T013418Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T013710Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T014158Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T014446Z.json`
- Runtime patch: `patches/minimax-qkvar-skipclone-fp32n2-20260518.patch`

## Decision

Promote as the current strict MiniMax baseline. The improvement is modest, but it is quality-preserving and repeatable across four benchmark repeats.

- LocalMaxxing id: `cmpbz7lyc004rpc019jburzqv`
