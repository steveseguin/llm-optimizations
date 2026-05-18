# MiniMax M2.7 No-Clone + Final-Hidden Clone Retie

Date: 2026-05-18

## Result

`VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` was re-tested on top of the current MiniMax MoE work-sharing FlashAttention/PIECEWISE strict baseline.

The candidate passed the full strict quality gate and produced a small numerical retie of the promoted baseline:

- Run 1: `80.219132` output tok/s, `106.958843` total tok/s
- Run 2: `81.363908` output tok/s, `108.485210` total tok/s
- Mean: `80.791520` output tok/s, `107.722027` total tok/s

The previous promoted LocalMaxxing result is `80.602755` output tok/s and `107.470340` total tok/s. This candidate is only `+0.23%` output tok/s over that result, so it is treated as a validated tie rather than a material new win.

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
LABEL=moe-ws-no-clone-clonefinal-flash-piecewise-strict-20260518 \
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
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1 \
VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1 \
VLLM_RUNTIME_REQUIRE_ANY_MARKERS=VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE,VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-moe-ws-no-clone-clonefinal-flash-piecewise-strict-20260518 \
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}' \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-no-clone-clonefinal-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T073625Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-no-clone-clonefinal-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T073625Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T075219Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T075513Z.json`

## Decision

Do not submit to LocalMaxxing as a new result. The result is honest and quality-preserving, but the delta is inside normal two-run variance. Keep it as evidence that allreduce no-clone plus final-hidden clone is safe enough to remain an optional ingredient for later collective-boundary experiments.
