# MiniMax Fast Async Output-List Patch Note

Date: 2026-05-17

## Intent

Test whether bypassing generic `Tensor.tolist()` for the strict async,
batch-1, greedy MiniMax path improves 4x B70 decode throughput.

## Default-Off Runtime Flag

```bash
export VLLM_XPU_FAST_ASYNC_OUTPUT_LIST=1
```

## Implementation Sketch

Files patched in the local vLLM source tree and installed venv:

- `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`

The patch adds a guarded branch in `AsyncGPUModelRunnerOutput.get_output()`:

```python
if (
    os.environ.get("VLLM_XPU_FAST_ASYNC_OUTPUT_LIST", "0") == "1"
    and self.sampled_token_ids_cpu.shape[0] == 1
    and self._logprobs_tensors_cpu is None
    and not self._invalid_req_indices
):
    with timed_region("gpu_model_runner.async_output_fast_list"):
        valid_sampled_token_ids = [[int(self.sampled_token_ids_cpu[0, 0])]]
else:
    with timed_region("gpu_model_runner.async_output_tolist"):
        valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
```

The installed venv also needed:

```python
from vllm.utils.xpu_decode_timing import timed_region
```

Harness updates:

- `scripts/bench-vllm-minimax-autoround-xpu.sh` logs
  `VLLM_XPU_FAST_ASYNC_OUTPUT_LIST`.
- `scripts/inspect-vllm-runtime.py` records the fast-list marker.

## Validation

Syntax check:

```bash
python3 -m py_compile \
  /home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py \
  /home/steve/llm-optimizations-publish/scripts/inspect-vllm-runtime.py

/home/steve/.venvs/vllm-xpu/bin/python -m py_compile \
  /home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py
```

Quality canary:

- raw145 n64 exact hash passed
- hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Benchmark:

- adjacent control: `61.290142` output tok/s, `81.720190` total tok/s
- fast list: `61.261073` output tok/s, `81.681430` total tok/s

## Decision

Reject as a performance promotion. The patch is quality-safe on the first exact
canary, but it is not faster than the adjacent control. Keep it default-off
only as a diagnostic/reference branch.
