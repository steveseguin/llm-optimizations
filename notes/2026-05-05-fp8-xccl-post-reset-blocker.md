# FP8 TP4 XCCL Post-Reset Blocker

Date: 2026-05-05

## Context

The strongest prior static FP8 candidate was:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- engine: patched vLLM/XPU, `quantization=compressed-tensors`;
- TP4, XPU FlashAttention2, n-gram speculative decode;
- `num_speculative_tokens=4`, lookup min/max `2/4`;
- shape: 512 prompt / 512 output;
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-in512-out512-bs1-20260504T230143Z.json`;
- result: `50.193 tok/s` output, `100.386 tok/s` total, two measured iterations.

That result is promising but was not submitted because it still needs a longer validation run. A previous three-iteration validation for lookup min/max `2/5` was submitted at `46.067 tok/s`.

## New Validation Attempt

Command shape:

```bash
MODEL_DIR=/home/steve/models/qwen3.6-27b-fp8-vrfai
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
TP=4
PP=1
INPUT_LEN=512
OUTPUT_LEN=512
MAX_MODEL_LEN=1024
GPU_MEM_UTIL=0.90
NUM_ITERS=3
WARMUP_ITERS=1
QUANTIZATION=compressed-tensors
SPECULATIVE_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
EXTRA_ARGS='--disable-log-stats --no-enable-prefix-caching'
```

Log:

`/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260505T010938Z.log`

Outcome:

- no JSON result;
- segfault during XCCL communicator initialization;
- stack entered oneCCL `coll_init`, then SYCL/UR Level Zero `urProgramBuildExp`;
- vLLM reported `Engine core initialization failed`.

## Standalone XCCL Smokes

After cleaning stale `/dev/shm` Python/vLLM segments and clearing `~/.cache/neo_compiler_cache`, standalone XCCL allreduce smokes still segfaulted before the first 4 KB measured allreduce:

- 4 ranks, default topology: rank 3 segfault;
- 4 ranks, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`: rank 0 segfault;
- 2 ranks, selector `0,3`: rank 1 segfault;
- 2 ranks, `CCL_ATL_TRANSPORT=mpi`: rank 1 segfault.

## Interpretation

FP8 tensor-parallel validation is blocked in this boot session by oneCCL/XCCL instability, not by vLLM model code or the FP8 checkpoint. The earlier valid submitted result remains `46.067 tok/s`. The `50.193 tok/s` candidate should be revalidated after a reboot or driver reload before public submission.

Do not run more TP vLLM benchmarks until a standalone XCCL allreduce smoke completes successfully.
