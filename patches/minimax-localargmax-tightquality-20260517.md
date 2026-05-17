# MiniMax Tight Quality Wrapper Patch

Date: 2026-05-17

This patch record describes the local script changes used for the strict MiniMax
local-argmax refresh accepted on LocalMaxxing as `cmp9xpe3w04pdo4013acdikt7`.
The full local patch is stored at:

`/home/steve/llm-optimizations-publish/patches/minimax-localargmax-tightquality-20260517.patch`

## Changed Files

- `prompts/minimax-arithmetic-canary-raw.txt`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`
- `scripts/bench-vllm-minimax-autoround-xpu.sh`
- `scripts/inspect-vllm-runtime.py`
- `README.md`
- `plans/2026-05-17-minimax-post-61-next-plan.md`

## Functional Changes

- Tightened the MiniMax arithmetic canary prompt to require exactly `42` and to reject quotes, symbols, punctuation, spaces, markdown, and explanations.
- Added prompt-scoped exact regex checks to the semantic, repeat-arithmetic, and extended quality suites:
  - `--require-prompt-regex '1:^\s*42\s*$'` in semantic and extended suites.
  - `--require-prompt-regex '0:^\s*42\s*$'` in the repeat-arithmetic suite.
- Added runtime diagnostics before quality and benchmark model loads through `scripts/inspect-vllm-runtime.py`.
- Added a quality startup guard so stuck model loads fail with a log tail instead of hanging indefinitely.
- Added benchmark-side runtime diagnostics and optional `VLLM_RUNTIME_REQUIRE_LOG_REGEX` enforcement.
- Extended `inspect-vllm-runtime.py` to inspect `vllm.v1.worker.gpu_model_runner` so future runs can require local-argmax control markers outside `logits_processor.py`.

## Reproduction Env

```bash
FI_TCP_IFACE=wlxe865d47e3a48 \
CCL_KVS_IFACE=wlxe865d47e3a48 \
USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2 \
VLLM_XPU_LOCAL_ARGMAX_DECODE=1 \
VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1 \
VLLM_BENCH_TEMPERATURE=0 \
VLLM_RUNTIME_EXPECT_LOGITS_PROCESSOR=/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py \
VLLM_RUNTIME_REQUIRE_MARKERS=logits.local_argmax_pair_all_gather \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=2 \
LABEL=minimaxlogits-localargmax-tightquality-extended-bench \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-strict-minimaxlogits-localargmax-tightquality-extended-20260517 \
scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Validation

- `jq empty` on the new structured data, LocalMaxxing payload, and response JSON.
- `bash -n` on `scripts/run-minimax-strict-quality-gated-candidate.sh` and `scripts/bench-vllm-minimax-autoround-xpu.sh`.
- `python3 -m py_compile` on `scripts/inspect-vllm-runtime.py` and `scripts/run-vllm-minimax-quality-check.py`.
- Full quality gate and two p512/n1536 benchmark repeats passed before LocalMaxxing submission.
