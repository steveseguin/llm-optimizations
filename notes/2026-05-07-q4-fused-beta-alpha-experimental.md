# Qwen35 Fused Beta/Alpha Projection Experiment

Date: 2026-05-07

## Summary

This branch adds optional Qwen35/Qwen35MoE support for a fused recurrent projection tensor, `blk.N.ssm_ba.weight`, generated from the existing `blk.N.ssm_beta.weight` and `blk.N.ssm_alpha.weight` tensors.

The original interleaved layout was speed-positive but failed the stronger TP3 token/logit guard. The corrected flat layout is quality-cleared against the original model only when the root-residual collective shortcut is disabled. It is a useful software direction, but not yet the global Q4_0 speed path because the current unsafe root-residual recipe is still faster on raw throughput.

## Why

The one-token tensor-family profile showed the recurrent F32 alpha/beta projections still visible after the larger Q4_0 and communication fusions. Combining those two projections removes one small F32 projection per recurrent layer.

## Implementation

- Script: `scripts/add-qwen35-fused-ba-gguf.py`
- Generated model: `/home/steve/models/qwen3.6-27b-q4_0-fused-ba-gguf/Qwen3.6-27B-Q4_0-fused-ba.gguf`
- Added fused tensors: `48`
- Fused layout: all beta rows followed by all alpha rows.
- Loader behavior: if `ssm_ba` exists, the separate alpha/beta tensors are loaded with fallback skip flags.
- Required split fix: Qwen35 fused `ssm_ba` must use the same per-group granularity as separate alpha/beta. Qwen3Next keeps the doubled granularity.
- Required meta fix: preserve `ssm_ba` split segments through `MUL_MAT` and exact axis-0 `VIEW` subsets so beta and alpha inherit the correct row ownership.

## Quality-Cleared Safe Result

TP3, `SYCL2/SYCL1/SYCL3`, tensor split `1/1/1`, `-ub 128`, `--poll 25`, f16 KV, flash attention, Q8 activation cache, single-kernel allreduce, fused allreduce+ADD, fused allreduce+GET_ROWS, fused MMVQ2, fused MMVQ2+SwiGLU, fused RMS_NORM+scale-MUL, and `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=0`:

| Model | Prompt | Decode | Computed total |
| --- | ---: | ---: | ---: |
| Original Q4_0 | `193.872714 tok/s` | `49.524990 tok/s` | `78.895931 tok/s` |
| Fused `ssm_ba` Q4_0 | `200.480796 tok/s` | `50.129900 tok/s` | `80.204735 tok/s` |

Delta versus the original under the same safe flags: `+0.604910 tok/s` decode, or `+1.22%`.

Artifacts:

- Full safe A/B: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/fused-ba-model/fused-ba-flat-no-root-p512n512-r3-20260507T065943Z.tsv`
- Short safe flag screen: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/fused-ba-model/fused-ba-flat-safe-flags-p0n128-r3-20260507T065456Z.tsv`
- Final no-root token/logit byte-compare: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-no-root-final-20260507T073839Z`

## Correctness Status

The final no-root probe matched the original model byte-for-byte:

- selected token: `321`
- logit hash: `4f77fb8d90ed49e6`
- root-residual: disabled
- meta allreduce+ADD: enabled
- meta allreduce+GET_ROWS: enabled

The flat layout also passed single-GPU and TP3 no-fusion token/logit checks before the full safe-stack probe:

- Single GPU: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-logit-probe-flat-single-20260507T061507Z`
- TP3 all fusions disabled: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-logit-probe-flat-tp3-nofusions-20260507T061721Z`

## Unsafe Interaction Found

The old root-residual path is not quality-cleared after the stronger token/logit probe. The minimal bad interaction is:

```bash
GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1
GGML_META_FUSE_ALLREDUCE_ADD=1
```

Each option passed individually, but the pair diverged. Artifact:

- `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-meta-pair-isolation-20260507T064644Z`

Observed failure:

- original token/hash: `220` / `c57597537930eab9`
- fused token/hash: `271` / `f8be73fee293172b`

Follow-up attempts:

- A host wait inside `comm_allreduce_add_tensor` made the all-current root path correct but collapsed throughput to about `29 tok/s`, so it was reverted.
- A root queue barrier did not fix correctness, so it was reverted.

## Decision

Keep the fused beta/alpha work as a quality-cleared no-root experimental branch and as evidence that small recurrent projection fusion is worth pursuing. Do not use `GGML_SYCL_COMM_FUSEADD_ROOT_RESIDUAL=1` for quality claims until the root/meta-add ordering bug is fixed by a lower-overhead synchronization or dependency model.

## LocalMaxxing

Submitted the reduced core-metric payload after the detailed payload hit the known API HTTP 500 path:

- ID: `cmov6p4r7007tqr01yi8ug4un`
- Status: `APPROVED`
- Quantization label: `Q4_0+fused-ssm_ba`
- Output throughput: `50.129900 tok/s`
- Total throughput: `80.204735 tok/s`

The notes explicitly mark this as an experimental augmented-GGUF result and state that root-residual is disabled because the root-residual plus meta allreduce-add interaction is not quality-cleared.

Next source-level work:

1. Fix the root-residual plus meta allreduce-add ordering hazard without a global host wait.
2. Consider making the beta/alpha fusion a runtime-side graph rewrite instead of requiring an augmented GGUF.
3. Add a focused recurrent projection equivalence harness so future layout changes do not need full-model token/logit probes first.
