# Qwen35 Fused Beta/Alpha Projection Experiment

Date: 2026-05-07

## Summary

This branch adds optional Qwen35/Qwen35MoE support for a fused recurrent projection tensor, `blk.N.ssm_ba.weight`, generated from the existing `blk.N.ssm_beta.weight` and `blk.N.ssm_alpha.weight` tensors.

The speed signal is positive but the quality guard is not cleared. Do not treat this as a LocalMaxxing-submittable quality-preserving result yet.

## Why

The one-token tensor-family profile showed the recurrent F32 alpha/beta projections still visible after the larger Q4_0 and communication fusions. Combining those two projections removes one small F32 projection per recurrent layer.

## Implementation

- Script: `scripts/add-qwen35-fused-ba-gguf.py`
- Generated model: `/home/steve/models/qwen3.6-27b-q4_0-fused-ba-gguf/Qwen3.6-27B-Q4_0-fused-ba.gguf`
- Added fused tensors: `48`
- Loader behavior: if `ssm_ba` exists, the separate alpha/beta tensors are loaded with fallback skip flags.
- Required split fix: Qwen35 fused `ssm_ba` must use the same per-group granularity as separate alpha/beta. The earlier doubled granularity produced a split mismatch at `ADD(alpha_cont, ssm_dt.bias)`.

## Results

TP3, current best Q4_0 stack, `--poll 25`, `-ub 128`:

| Shape | Original | Fused `ssm_ba` |
|---|---:|---:|
| `p0/n128/r3` decode | `48.726023 tok/s` | `50.414495 tok/s` |
| `p512/n512/r3` prompt | `195.046919 tok/s` | `200.681918 tok/s` |
| `p512/n512/r3` decode | `50.914445 tok/s` | `51.338431 tok/s` |
| `p512/n512/r3` computed total | `80.750127 tok/s` | `81.760817 tok/s` |

Artifacts:

- Short A/B: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/fused-ba-model/fused-ba-layout2-ab-p0n128-r3-20260507T053025Z.tsv`
- Full A/B: `/home/steve/bench-results/qwen36-q4_0-gguf/tp3-refresh-20260507/fused-ba-model/fused-ba-layout2-ab-p512n512-r3-20260507T054539Z.tsv`
- Logit probe: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-logit-probe-20260507T052727Z`
- PPL attempt: `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/fused-ba-ppl-20260507T053311Z`

## Quality Status

- Token/logit probe step 0 selected the same top token (`271`) for original and fused.
- Full logit hashes and top-logit values differed at step 0.
- `llama-completion` stdout byte-compare is not reliable here because two original-model runs also differed.
- `llama-perplexity --chunks 4 -c 512` did not finish the original model inside `600 s`; the fused follow-up was killed rather than spending another 10 minutes.

## Decision

Keep this as an experimental speed branch. Do not submit to LocalMaxxing until either:

1. a lower-level recurrent projection harness proves row/layout equivalence, or
2. a practical quality proxy clears the fused model against the original.

The next source-level direction is preserving row-level segment ownership through the fused reshape/view chain instead of only matching per-device segment totals.
