# MiniMax M2.7 Strict Quality Gate and TP Boundary Candidates

Date: 2026-05-16

Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM 0.20.1-local XPU with llm-scaler INT4 MoE path

## Why

The current accepted MiniMax result is quality-valid at roughly 65.75 output
tok/s, but earlier optimization attempts showed that a fast graph path can
silently corrupt output. This pass makes the pre-benchmark gate stricter before
testing more TP communication-boundary changes.

New artifacts:

- `prompts/minimax-raw145-tokenhash-canary.txt`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`
- `data/minimax-m27-strict-quality-gate-candidates-20260516.json`

## Strict Gate

The strict gate matches the accepted piecewise XPU graph recipe:

- TP4, `float16`, context 2048, MBT 512, block size 256.
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`

It requires:

1. Raw145, 64 generated tokens, exact combined token hash:
   `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
2. Raw145, 256 generated tokens, exact combined token hash:
   `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
3. A two-repeat semantic suite that must be deterministic:
   `PASS`, `42`, and `def add_one(x): return x + 1`.

The baseline strict validation passed:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-strict-baseline-strict-tp4-ctx2048-mbt512-bs256-20260516T000402Z-summary.json`
- Semantic suite hash:
  `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`

## Candidate: MoE Delayed Allreduce

Env:

```bash
VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1
```

Result:

- Raw145 64-token exact hash: pass.
- Raw145 256-token exact hash: pass.
- Semantic suite: fail, nondeterministic greedy token hashes.
- Difference: arithmetic canary emitted `42` in one run and ` 42` in the next.
- Benchmark: not run.

Artifacts:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-delay-allreduce-strict-tp4-ctx2048-mbt512-bs256-20260516T001127Z-summary.json`
- Semantic failure:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-delay-allreduce-strict-tp4-ctx2048-mbt512-bs256-20260516T001127Z-quality/semantic-suite-n64-r2.json`

Decision: reject. Do not promote or submit to LocalMaxxing.

## Candidate: Distributed Residual Allreduce

Env:

```bash
VLLM_MINIMAX_M2_DIST_RESIDUAL_ALLREDUCE=1
```

Result:

- Raw145 64-token exact hash: pass.
- Raw145 256-token exact hash: pass.
- Semantic suite: fail, nondeterministic greedy token hashes.
- Difference: arithmetic canary emitted `42` in one run and ` 42` in the next.
- Benchmark: not run.

Artifacts:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-dist-residual-allreduce-strict-tp4-ctx2048-mbt512-bs256-20260516T002241Z-summary.json`
- Semantic failure:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-dist-residual-allreduce-strict-tp4-ctx2048-mbt512-bs256-20260516T002241Z-quality/semantic-suite-n64-r2.json`

Decision: reject. Do not promote or submit to LocalMaxxing.

## Candidate: MiniMax Logits MoE Path

Env:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1
```

Result:

- Raw145 64-token exact hash: pass.
- Raw145 256-token exact hash: pass.
- Semantic suite: fail, nondeterministic greedy token hashes.
- Difference: arithmetic canary emitted `42` in one run and ` 42` in the next.
- Benchmark: not run.

Artifacts:

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-strict-tp4-ctx2048-mbt512-bs256-20260516T004653Z-summary.json`
- Semantic failure:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-logits-strict-tp4-ctx2048-mbt512-bs256-20260516T004653Z-quality/semantic-suite-n64-r2.json`

Decision: reject. The intended `moe_forward_tiny_cutlass_nmajor_int4_u4_minimax`
logits path was exercised, but it failed the semantic determinism gate. Do not
promote or submit to LocalMaxxing.

## Lessons

The raw exact canary is necessary but not sufficient. Both rejected candidates
preserved the long repetitive raw canary but introduced nondeterministic greedy
output on a simple arithmetic canary. From here, any speed claim needs both
exact corruption canaries and deterministic semantic canaries before benchmark
numbers are considered.

Next speed work should focus on deterministic device-level changes rather than
Python-level collective boundary rearrangements: real XPU fused allreduce plus
RMS/epilogue kernels, lower-level XCCL/Level Zero timing, and MoE kernel
profiling that does not alter residual/allreduce ordering.

The rejected MiniMax logits path reinforces the same rule: matching long-form
token hashes is not enough if short semantic canaries are not deterministic. It
may still be useful as a source reference for future XPU kernel work, but not as
a runtime option for quality-preserving submissions.
