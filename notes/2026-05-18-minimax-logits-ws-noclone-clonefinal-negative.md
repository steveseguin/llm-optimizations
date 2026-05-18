# MiniMax Logits-WS No-Clone + Final-Hidden Clone Negative

Date: 2026-05-18

## Result

`VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` plus
`VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` was tested on top of the current
MiniMax logits-to-work-sharing strict baseline.

The candidate passed the full strict quality gate, but it was slower than the
promoted logits-WS baseline:

- Run 1: `80.017396` output tok/s, `106.689861` total tok/s
- Run 2: `82.024853` output tok/s, `109.366471` total tok/s
- Mean: `81.021124` output tok/s, `108.028166` total tok/s

The promoted logits-WS baseline is `81.758267` output tok/s and `109.011023`
total tok/s. This branch is `-0.90%` output tok/s, so it is not promoted and
was not submitted to LocalMaxxing.

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
LABEL=minimax-logits-ws-noclone-clonefinal-strict-20260518 \
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
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1 \
VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1 \
VLLM_RUNTIME_REQUIRE_LOG_REGEX='Using llm-scaler XPU INT4 MiniMax logits WS decode path' \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-logits-ws-noclone-clonefinal-strict-20260518 \
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}' \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-ws-noclone-clonefinal-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T085310Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-ws-noclone-clonefinal-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T085310Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T090855Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T091152Z.json`

## Decision

Do not promote and do not submit to LocalMaxxing. The quality result is useful:
it shows that the clone-removal flag is not a free win once the exact MiniMax
router logits are feeding the work-sharing MoE kernel. The next useful work is
to label and time the residual allreduce/final-logits boundaries, then choose a
narrow fusion target from measured decode cost instead of further flag reties.
