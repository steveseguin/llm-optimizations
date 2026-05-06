# FP8 PP2xTP2 post-reboot validation

Date: 2026-05-06

## Goal

Re-check the four-rank vLLM/XPU path after reboot without changing power limits or model quality. This was a stability and plumbing validation, not a leaderboard run.

Runtime rule kept: do not source oneAPI `setvars.sh` for vLLM. Keep `/home/steve/.venvs/vllm-xpu-managed/lib` first in `LD_LIBRARY_PATH`, use `CCL_ATL_TRANSPORT=ofi`, and set `PYTHONPATH=/home/steve/src/vllm`.

## XCCL Gate

Command used four ranks with `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`.

Result: pass.

- all ranks printed `init ok`
- all ranks printed `barrier ok`
- all ranks printed `allreduce ok 4.0`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/xccl-4rank-pp2tp2-gate-20260506T212704Z.log`

## PP2xTP2 Non-Spec

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`
- topology: `TP=2`, `PP=2`
- quantization: `compressed-tensors`
- prompt/output: `512/32`
- `MAX_MODEL_LEN=1024`
- `GPU_MEM_UTIL=0.80`
- `WARMUP_ITERS=0`, `NUM_ITERS=1`

Result:

- completed successfully
- average latency: `22.95836220899946 s`
- computed output throughput: `1.3938276480129885 tok/s`
- computed total throughput: `23.695070016220807 tok/s`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T212722Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T212722Z.log`

The short output length makes this a load/plumbing check only. Existing longer PP2xTP2 measurements remain more representative.

## PP2xTP2 CPU n-gram

Configuration: same model and topology, plus:

```json
{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":4}
```

Additional env:

- `VLLM_XPU_TRACE_NGRAM_PP=1`
- `VLLM_BENCH_LATENCY_PROMPT_MODE=repeat`
- `VLLM_BENCH_LATENCY_REPEAT_PERIOD=16`
- `VLLM_BENCH_LATENCY_PROMPT_SEED=123`

Result:

- completed successfully
- average latency: `20.274570381006924 s`
- computed output throughput: `1.5783318412496363 tok/s`
- computed total throughput: `26.831641301243817 tok/s`
- trace lines matching n-gram/speculative markers: `335`
- no old GDN assertion or device-lost failure observed
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T213059Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out32-bs1-20260506T213059Z.log`

Important finding: the PP2 n-gram path still proposed zero draft tokens. Logs repeatedly show `draft=[0]`, `draft_lens=[0]`, and `spec_lens={}`. This confirms the guard path is stable after reboot, but it does not provide actual speculative speedup.

## Interpretation

The post-reboot FP8 communication stack is healthy. PP2xTP2 can load and generate, and the patched speculative plumbing no longer hits the earlier assertion in this short run. It still does not produce useful draft tokens in PP2, so TP4/PP1 remains the Qwen3.6 27B FP8 speed path.

No LocalMaxxing submission: these are stability diagnostics, not valid performance improvements.
