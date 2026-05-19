# MiniMax Source-Path GPUModelRunner Sync Patch

Status: accepted as a source-path repair on 2026-05-19.

## Code Touches

- `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py`

The goal was not a new speed optimization. It was to make source-tree execution
match the accepted installed venv behavior again after a strict source-path run
failed `raw145-n256-exact`.

## Restored Runtime Shape

Restored device-aware async-copy helpers:

```python
def _new_copy_stream(device_type: str):
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.Stream()
    return torch.cuda.Stream()


def _new_copy_event(device_type: str):
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.Event()
    return torch.Event()


def _current_copy_stream(device_type: str):
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.current_stream()
    return torch.cuda.current_stream()


def _copy_stream_context(stream, device_type: str):
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu.stream(stream)
    return torch.cuda.stream(stream)
```

Applied those helpers in:

- `AsyncGPUModelRunnerOutput`
- `AsyncGPUPoolingModelRunnerOutput`
- `GPUModelRunner.__init__`
- `GPUModelRunner._get_or_create_async_output_copy_stream`

Restored the optional XPU sampled-token clone guard:

```python
if (
    self.use_async_scheduling
    and os.environ.get("VLLM_XPU_ASYNC_CLONE_SAMPLED_TOKEN_IDS", "0") == "1"
    and sampler_output.sampled_token_ids.device.type == "xpu"
):
    sampler_output.sampled_token_ids = sampler_output.sampled_token_ids.clone()
```

Removed broad `timed_region(...)` wrappers around the hot compiled decode,
postprocess, logits, sampling, and update-state sections that had drifted into
the source copy. Disabled diagnostics must stay out of the compiled path unless
they are proven TorchDynamo-compatible and performance-neutral.

## Verification

Post-patch:

```bash
diff -q /home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py \
  /home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py
```

returned clean, and the source file hash was:

```text
d900ae2c37df0c70df42bb33b350dd4831f0399ccc7222f7b7e2f951efe9308f
```

Strict source-path quality screen passed:

- raw145 n64 exact
- raw145 n256 exact
- semantic suite

Full strict repeat then passed:

- raw145 n64 exact
- raw145 n256 exact
- semantic suite
- arithmetic repeat n64 r8
- extended sixpack n64 r2

Benchmark mean was `87.976305` output tok/s / `117.301741` total tok/s across
two p512/n1536 repeats. This is below the current clean high and was not
submitted to LocalMaxxing.
