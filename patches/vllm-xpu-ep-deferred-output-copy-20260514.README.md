# vLLM XPU EP Deferred Output Copy Patch Snapshot

Date: 2026-05-14

Purpose: reproduce the MiniMax M2.7 expert-parallel diagnostic branch that
combines the XPU padded uneven `all_gatherv` workaround with a deferred
sampled-token output-copy experiment.

Contents:

- `vllm-xpu-ep-deferred-output-copy-20260514.patch.gz.b64`: compressed patch
  against `/home/steve/src/vllm` for the active EP diagnostics in
  `xpu_communicator.py` and `gpu_model_runner.py`.

Reapply example:

```bash
base64 -d patches/vllm-xpu-ep-deferred-output-copy-20260514.patch.gz.b64 | \
  gunzip | git -C /path/to/vllm apply
```

Outcome:

The padded uneven `all_gatherv` workaround remains useful for the synthetic
XCCL repro, but the deferred output-copy branch does not make MiniMax EP
benchmarkable. A short EP probe still triggers `UR_RESULT_ERROR_DEVICE_LOST`
during sampled-token synchronization, creates xe devcoredumps, and requires
`xpu-smi config --reset` before further XPU work.
