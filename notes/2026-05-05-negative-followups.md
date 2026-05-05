# 2026-05-05 Negative Follow-Ups

This note records the follow-up screens that did not beat the current best B70 paths. These are kept because they narrow the remaining search space and identify backend bugs.

## Qwen3.6 FP8 TP2/PP2

- Wrapper change: `bench-vllm-qwen36-fp8.sh` now accepts `PP=<n>` and passes `--pipeline-parallel-size`.
- TP2/PP2 no-spec cold smoke, `64/16`: `0.772 tok/s` output. This was dominated by compile/cold-start and pipeline bubble.
- TP2/PP2 no-spec warm-cache screen, `64/64`: `27.795 tok/s` output.
- TP2/PP2 n-gram speculative decode, `512/128`, `num_speculative_tokens=4`, lookup `2/5`: failed during warmup with `AttributeError: 'XPUModelRunner' object has no attribute 'drafter'`, then the worker pool hung and was killed.
- Decision: PP2 is a possible capacity fallback for larger models, but not a Qwen3.6 27B single-stream speed path. PP2+n-gram has a vLLM/XPU bug.

## Qwen3.6 FP8 TP4 oneCCL Topology Override

- Tested `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` because oneCCL repeatedly logs that this bypass is available.
- Command shape: TP4, static FP8, n-gram speculative decode with 4 draft tokens, lookup `2/5`, `512/256`.
- Result: `40.049 tok/s` output.
- Baseline same shape with default topology recognition: `42.245 tok/s` output.
- Decision: keep default oneCCL topology recognition.

## Qwen3.6 Q4_0 GGUF Small-F32 Allreduce

- Extended the env-gated `GGML_SYCL_COMM_SMALL_F32=1` path to 4 backends for a controlled screen.
- 4x B70 selector `0,1,2,3`, `512/128`, event barrier on: `31.763 tok/s`, below prior 4x event-barrier `32.427 tok/s`.
- 3x B70 selector `2,1,3`, `512/128`, event barrier on: `34.874 tok/s`, far below the 3x event-barrier path without small-F32.
- Decision: fixed 256-work-item small-F32 allreduce underutilizes B70 and should stay disabled. Q4 needs fused matmul/reduction work, not more tiny-kernel launch variants.

## MiniMax M2.7 MUL_MAT_ID Guard

- Added a diagnostic log before the existing split-buffer `MUL_MAT_ID` assert.
- Changed SYCL `supports_op` to reject `GGML_OP_MUL_MAT_ID` when `src0` is already a SYCL split buffer.
- `-ncmoe 13`: failure moved from the `129761280` byte split expert allocation to a monolithic fallback allocation: `unable to allocate SYCL1 buffer` of `26877100032` bytes.
- `-ncmoe 50`: fallback failed with `unable to allocate SYCL3 buffer` of `20157825024` bytes.
- Decision: the guard is diagnostic only. MiniMax needs real split-buffer `MUL_MAT_ID` or expert-owned execution; fallback placement is too coarse.

## No LocalMaxxing Submission

None of these screens were submitted because they are regressions, smoke tests, or failed backend paths rather than useful leaderboard results.
