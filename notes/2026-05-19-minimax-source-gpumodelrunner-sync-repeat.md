# MiniMax Source-Path GPUModelRunner Sync Repeat

Date: 2026-05-19

## Goal

Restore the local source-tree execution path after a source-path repeatability
run failed the `raw145-n256-exact` gate. The accepted MiniMax clean high was
measured through the installed vLLM package, while the follow-up repeat loaded
`/home/steve/src/vllm` through `PYTHONPATH`.

## Root Cause

`vllm/v1/worker/gpu_model_runner.py` in the source tree had drifted from the
installed venv copy. The drift included XPU async-copy stream/event handling and
extra timing wrappers around decode/postprocess paths. That source-path run
passed raw145 n64 but failed raw145 n256 with a deterministic different token
hash, so it was not a valid reproducibility result.

## Repair

Restored the source-tree `gpu_model_runner.py` to match the accepted installed
copy for the relevant runtime behavior:

- XPU-aware async output copy stream/event helpers.
- Pooling/output copy paths using the XPU helper wrappers.
- Async scheduling stream/event construction through the device-aware helpers.
- `VLLM_XPU_ASYNC_CLONE_SAMPLED_TOKEN_IDS` clone guard after sampling.
- Removal of broad decode/postprocess timing wrappers from the compiled path.

Post-repair file check:

```text
gpu_model_runner.py sha256: d900ae2c37df0c70df42bb33b350dd4831f0399ccc7222f7b7e2f951efe9308f
```

The repaired source copy matched the installed venv copy for:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/models/minimax_m2.py`
- `vllm/model_executor/layers/quantization/moe_wna16.py`

## Validation

Quality screen:

- `raw145-n64-exact`: passed
- `raw145-n256-exact`: passed
- `semantic-suite-n64-r2`: passed

Full strict repeat:

- `raw145-n64-exact`: passed
- `raw145-n256-exact`: passed
- `semantic-suite-n64-r2`: passed
- `arithmetic-repeat-n64-r8`: passed
- `extended-sixpack-n64-r2`: passed

Benchmark result, p512/n1536, ctx2048, TP4:

- Output tok/s samples: `87.55516060057364`, `88.39745039435884`
- Mean output tok/s: `87.97630549746624`
- Total tok/s samples: `116.7402141340982`, `117.86326719247846`
- Mean total tok/s: `117.30174066328833`

## Decision

Do not submit this run to LocalMaxxing. It is a useful reproducibility repair
and quality confirmation, but it does not exceed the current clean direct Q/K
variance high of `88.501953` output tok/s / `118.002604` total tok/s.

## Notes

The run still occasionally logs nonfatal Intel compiler failures for generated
Triton kernels:

```text
ocloc failed with error code 245
IGC: Internal Compiler Error: Floating point exception
```

Those warnings did not cause quality failure in this run, but they remain useful
evidence that some graph shapes are hitting immature BMG compiler paths.
