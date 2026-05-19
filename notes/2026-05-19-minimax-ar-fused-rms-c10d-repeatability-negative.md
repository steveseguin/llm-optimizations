# MiniMax AR+RMS C10D Repeatability Rejection, 2026-05-19

## Result

Rejected. The MiniMax post-attention AR+RMS fused helper is not repeatable
enough to promote.

The candidate passed an initial strict quality screen after fixing the process
group name, but a fresh speed-screen label failed the `raw145-n256-exact` token
hash before any benchmark repeats ran. That is a quality failure for our
purposes, even though the generated text was nontrivial and deterministic
inside that single run.

## Candidate

Flag:

```bash
VLLM_MINIMAX_AR_FUSED_RMS_XPU=1
```

The MiniMax layer path skips the normal `RowParallelLinear` reduction and calls:

```python
torch.ops.minimax_ar_fused_rms_xpu.ar_fused_add_rms(
    hidden_states,
    residual,
    self.post_attention_layernorm.weight.data,
    get_tp_group().device_group.group_name,
    self.post_attention_layernorm.variance_epsilon,
)
```

The process group detail matters:

- `get_tp_group().unique_name` is for vLLM custom all-reduce lookup, for example
  `tp:0`.
- `get_tp_group().device_group.group_name` is the c10d registered process group
  name required by `torch/csrc/distributed/c10d/Functional.hpp`.

Using `unique_name` failed with:

```text
RuntimeError: Could not resolve the process group registered under the name tp:0
```

Changing to `device_group.group_name` fixed the runtime failure, but did not make
the candidate reliable enough.

## Quality Evidence

Initial screen:

- Label: `minimax-ar-fused-rms-xpu-c10dgroup-quality-screen-20260519`
- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ar-fused-rms-xpu-c10dgroup-quality-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T084947Z-summary.json`
- Passed: `raw145-n64-exact`, `raw145-n256-exact`, `semantic-suite-n64-r2`

Repeat/speed screen:

- Label: `minimax-ar-fused-rms-xpu-speed-screen-20260519`
- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-ar-fused-rms-xpu-speed-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T090033Z-summary.json`
- Failed: `raw145-n256-exact`
- Failure reason: `combined token hash mismatch`
- Combined token hash: `19270162ff1caa83c1d62f8edbe8350ac9e3ec39f9a1e9d5880b1a15e3e0c00a`
- Combined text hash: `3c8e5adb9ecd6d17b8ef244a79e7e6aa2d3c3d8fa3b3ef90932a99eafc6f8b0d`

The speed screen stopped at quality failure, so there is no promoted throughput
number and no LocalMaxxing submission.

## Decision

Do not promote. Do not submit to LocalMaxxing.

Current clean high remains:

- `88.501953` output tok/s
- `118.002604` total tok/s
- LocalMaxxing: `cmpc8cmqm0060pc016g5l5ukh`

The AR+RMS boundary is still a plausible optimization target, but this
implementation shape is not acceptable. If we revisit it, the next version
should either preserve vLLM's proven all-reduce semantics and only fuse a safe
post-reduce epilogue, or add a true compiler/device lowering that is exact
across fresh graph captures.

## Artifacts

- Local data: `data/minimax-m27-ar-fused-rms-c10d-repeatability-negative-20260519.json`
- Extension source: `experiments/minimax_ar_fused_rms_xpu/`
- Smoke script: `benchmarks/b70_minimax_ar_fused_rms_op_smoke.py`
