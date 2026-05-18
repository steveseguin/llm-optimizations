# 2026-05-18 MiniMax Clone-Safe Custom Allreduce MBT768 Rejection

## Scope

This records a follow-up test of `MAX_BATCHED_TOKENS=768` on top of the current promoted MiniMax M2.7 AutoRound recipe:

- 4x Intel Arc Pro B70 32GB
- vLLM `0.20.1-local`, XPU TP4
- AutoRound INT4 W4A16 weights, FP16 activations
- XPU FlashAttention v2, XPU PIECEWISE graph
- exact MiniMax router logits into the llm-scaler INT4 MoE work-sharing decode path
- clone-safe compiled custom allreduce via `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` and `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`

The goal was to see whether MBT768 could improve the promoted clone-safe custom-allreduce result of `87.279129` output tok/s without changing output quality.

## Candidate

Key env:

```bash
MAX_BATCHED_TOKENS=768
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
CCL_TOPO_P2P_ACCESS=1
```

Runtime shape:

- Prompt/output: p512/n1536 benchmark, only after quality pass
- Context: 2048
- Batch: 1
- Block size: 256
- MBT: 768

## Result

Status: rejected before benchmark.

Summary:

`/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-clone-mbt768-ar-20260518-strict-tp4-ctx2048-mbt768-bs256-20260518T230121Z-summary.json`

Quality gate outcomes:

| Gate | Result |
| --- | --- |
| raw145 n64 exact | passed |
| raw145 n256 exact | passed |
| semantic suite n64 r2 | passed |
| arithmetic repeat n64 r16 | passed |
| extended sixpack n64 r2 | failed |

The failure was nondeterministic token output on the extended sixpack sort/list prompt. The semantic required substrings still appeared, but prompt 4 produced two different token/text hashes across greedy repeats:

- token hash run 0: `a768eeb66e15fe912fc035532db3a123ae75830fd20765abf2166603f43d3954`
- token hash run 1: `aa86a2578614dd7105f75c56c665c9bd776679d53371df8351be6f9984e93f5e`

Because repeatability is part of the quality contract, this candidate was not benchmarked and was not submitted to LocalMaxxing.

## Compiler/runtime notes

The first cold compile also emitted Intel compiler failures while compiling Triton reduction kernels:

- `Triton compilation failed: triton_red_fused__to_copy_mm_t_10`
- `ocloc` exit status 245
- `IGC: Internal Compiler Error: Floating point exception`

Execution continued after those compiler errors, but the later extended-sixpack nondeterminism makes this MBT768 graph boundary unsafe under the clone-safe custom-allreduce recipe.

## Decision

Keep the promoted clone-safe custom-allreduce recipe at `MAX_BATCHED_TOKENS=512`.

Do not retest MBT768 unchanged for this promoted path. If MBT/precompile ranges are revisited, they need a runtime/compiler change or a stricter graph-capture lifetime fix first.
