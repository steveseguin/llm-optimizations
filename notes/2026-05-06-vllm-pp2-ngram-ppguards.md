# 2026-05-06 - vLLM FP8 PP2 n-gram PP guards

## Context

Target: Qwen3.6 27B FP8, vLLM XPU, 4x Arc Pro B70, `TP=2`, `PP=2`, `INPUT_LEN=512`, `MAX_MODEL_LEN=1024`, batch 1.

The active benchmark wrapper imports the installed venv package at:

`/home/steve/.venvs/vllm-xpu-managed/lib/python3.12/site-packages/vllm`

The source checkout was also patched for reproducibility:

`/home/steve/src/vllm`

## What changed

Patch: `patches/vllm-pp2-ngram-ppguards-active-20260506.patch`

- Added an empty sampled-token guard for non-last PP ranks. The crash was:
  `IndexError: list index out of range` at `req_state.output_token_ids.append(new_token_ids[-1])`.
- Added a scheduler recovery guard for invalid negative token scheduling deltas. The observed bad scheduler output had `num_scheduled_tokens=-3`.
- Added a PP ownership guard for `ngram_gpu` buffers. Only the last PP rank allocates `token_ids_gpu_tensor`, but `_update_states()` was trying to update it on PP0.
- Added temporary trace logging behind `VLLM_XPU_TRACE_NGRAM_PP=1`.

## Results

CPU n-gram, before patch:

- Repro: `TP=2 PP=2`, `method=ngram`, 512 prompt / 128 output.
- Failed with `IndexError` in PP0. After the empty-token guard, it exposed invalid negative scheduling (`total_num_scheduled_tokens=-3`).

CPU n-gram, after patch:

- Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260506T075800Z.log`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out128-bs1-20260506T075800Z.json`
- Completed: `7.302738496 s` for 128 output tokens.
- Computed output speed: `17.527671 tok/s`.
- Trace showed `spec_lens={}` throughout, so CPU n-gram was effectively not drafting under this PP path.

GPU n-gram:

- Repro: `TP=2 PP=2`, `method=ngram_gpu`, 512 prompt / 64 output.
- First failed with `AttributeError: 'XPUModelRunner' object has no attribute 'token_ids_gpu_tensor'` on PP0.
- After the ownership guard, it got past the attribute error but repeatedly trimmed `total=5->1`, `spec_lens=4->{}`, `valid_counts=0`, then stalled with shared-memory broadcast warnings.
- Log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out64-bs1-20260506T080725Z.log`

## Interpretation

The PP2 speculative paths are now better understood but not performance-useful yet:

- `method=ngram` is stable after guards, but no speculative tokens are scheduled in this benchmark.
- `method=ngram_gpu` creates draft slots, but all drafts are reported invalid and trimmed, then the engine stalls.
- The strong FP8 path remains TP4 with CPU n-gram, previously measured at `49.581893 tok/s` for 512/512.

## Next

- Keep PP2 FP8 as a memory-capacity path, not a speed path, until speculation works under PP.
- For speed, keep optimizing TP4 FP8 and Q4 SYCL tensor-split paths.
- If returning to PP speculation, instrument why `ngram_gpu` valid counts are always zero and why PP CPU n-gram does not schedule draft tokens.
