# MiniMax M2.7 Logits-To-WS Strict Win

Date: 2026-05-18

## Result

`VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1` was added as a default-off MiniMax decode path. It keeps the exact MiniMax M2 routing rule from router logits, then feeds the existing llm-scaler unsigned INT4 work-sharing MoE kernel instead of returning through the older non-work-sharing logits path.

The candidate passed the full strict quality gate and produced a small new best:

- Run 1: `82.426806` output tok/s, `109.902408` total tok/s
- Run 2: `81.089729` output tok/s, `108.119638` total tok/s
- Mean: `81.758267` output tok/s, `109.011023` total tok/s

The previous promoted LocalMaxxing result was `80.602755` output tok/s and `107.470340` total tok/s. This is `+1.43%` output tok/s. The gain is not large, but it is above the recent retie noise and preserves quality, so it is promoted as the current best.

## Quality

All strict gates passed before benchmarking:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: pass
- arithmetic repeat: 16/16 deterministic exact `42`
- extended sixpack: pass

## Command

```bash
env -u VLLM_MINIMAX_MOE_DELAY_ALLREDUCE -u VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION -u VLLM_MINIMAX_QK_NORM_C10D_GROUP_NAME \
LABEL=minimax-logits-ws-flash-piecewise-strict-20260518 \
ATTENTION_BACKEND=default \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=2 \
QUALITY_TIMEOUT=35m \
BENCH_TIMEOUT=30m \
SHM_STALL_MAX_WARNINGS=3 \
MAX_BATCHED_TOKENS=512 \
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1 \
VLLM_RUNTIME_REQUIRE_LOG_REGEX='Using llm-scaler XPU INT4 MiniMax logits WS decode path' \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-logits-ws-flash-piecewise-strict-20260518 \
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}' \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Code Changes

- `custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`: added `moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws`, which computes exact MiniMax top-8 sigmoid+bias routing and then calls the work-sharing u4 MoE implementation.
- `custom_esimd_kernels_vllm/ops.py` and `__init__.py`: exported the new Python wrapper.
- `vllm/model_executor/layers/quantization/moe_wna16.py`: added the `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS` selector and log marker.
- `run-minimax-strict-quality-gated-candidate.sh`: records the selector in summary JSON.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-ws-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T080946Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-ws-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T080946Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T082531Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T082819Z.json`

## Decision

Promote as the current strict MiniMax baseline and submit to LocalMaxxing.

- LocalMaxxing id: `cmpay7th600bbmn01v6csyaro`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.
