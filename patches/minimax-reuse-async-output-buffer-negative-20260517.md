# MiniMax Reusable Async Output-Copy Buffer Patch Note

Date: 2026-05-17

## Intent

Test whether avoiding per-token CPU tensor allocation in
`AsyncGPUModelRunnerOutput` improves MiniMax M2.7 AutoRound INT4 decode on
4x B70.

## Default-Off Runtime Flags

```bash
export VLLM_XPU_REUSE_ASYNC_OUTPUT_COPY_BUFFER=1
export VLLM_XPU_ASYNC_OUTPUT_COPY_BUFFER_SLOTS=3
```

## Implementation Sketch

Files patched in the local vLLM source tree and installed venv:

- `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py`

The patch adds an optional ring of pinned CPU tensors:

```python
self.async_sampled_token_ids_pinned_cpu_buffers = [
    torch.empty(
        (self.max_num_reqs, 1),
        dtype=torch.int32,
        device="cpu",
        pin_memory=self.pin_memory,
    )
    for _ in range(num_async_output_buffers)
]
```

When enabled and the output has no logprobs, `AsyncGPUModelRunnerOutput` copies
`sampled_token_ids` into the next reusable CPU buffer on the async output copy
stream instead of allocating a new CPU tensor via `.to("cpu", non_blocking=True)`.

Harness updates:

- `scripts/bench-vllm-minimax-autoround-xpu.sh` logs the reusable-buffer flags.
- `scripts/inspect-vllm-runtime.py` records the reusable-buffer marker.

## Validation

Syntax check:

```bash
/home/steve/.venvs/vllm-xpu/bin/python -m py_compile \
  /home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py \
  /home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py \
  /home/steve/llm-optimizations-publish/scripts/inspect-vllm-runtime.py
```

Quality canary:

- raw145 n64 exact hash passed
- hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Benchmark:

- adjacent control: `61.626778` output tok/s, `82.169037` total tok/s
- reusable buffer: `61.200924` output tok/s, `81.601232` total tok/s

## Decision

Reject as a performance promotion. The patch is quality-safe on the first exact
canary but slower than the adjacent control and promoted strict baseline. Keep
it default-off only as a reference while pursuing a larger GPU-resident sampler
or scheduler handoff change.
