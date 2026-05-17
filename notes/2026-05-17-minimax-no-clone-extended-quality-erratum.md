# MiniMax No-Clone Extended Quality Erratum

Date: 2026-05-17

## Summary

The `66.757143` output tok/s MiniMax M2.7 AutoRound result should be treated as
a limited-canary performance datapoint, not a fully quality-promoted result.
It passed the initial raw145, semantic, and arithmetic-repeat checks before
benchmarking, but later extended retests found reproducibility and output
integrity failures.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Shape: p512 / n1536 / batch 1 / context 2048
- Candidate flag: `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- LocalMaxxing row to interpret cautiously: `cmp9tk7co04m3o401lhm2n9gm`

## Failed Retests

Cold-cache extended retest:

- Label: `no-clone-oldenv-extended-quality-repro`
- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-oldenv-extended-quality-repro-strict-tp4-ctx2048-mbt512-bs256-20260517T134254Z-summary.json`
- Status: `quality_failed_raw145_n64`
- Failure: raw145 n64 exact token hash mismatch after compiling a fresh cache
- Expected token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Actual token hash:
  `7f4222437a76869f7fde3e202d3c90a5e202583f603b57d4c9950fae6ad8bd67`
- NUL/control counts: `0` / `0`

Existing-cache extended retest:

- Label: `no-clone-oldenv-existingcache-extended-quality`
- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-oldenv-existingcache-extended-quality-strict-tp4-ctx2048-mbt512-bs256-20260517T134910Z-summary.json`
- Status: `quality_failed_extended_suite`
- Failure: extended sixpack had nondeterministic tokens, a degenerate NUL/control
  output, and a missing prompt-scoped required substring
- Extended sixpack artifact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-oldenv-existingcache-extended-quality-strict-tp4-ctx2048-mbt512-bs256-20260517T134910Z-quality/extended-sixpack-n64-r2.json`
- NUL/control counts: `64` / `64`

Async-scheduling-disabled retest:

- Label: `no-clone-asyncoff-extended-runtimeguard`
- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-asyncoff-extended-runtimeguard-strict-tp4-ctx2048-mbt512-bs256-20260517T140217Z-summary.json`
- Status: `quality_failed_repeat_arithmetic_suite`
- Failure: arithmetic repeat suite diverged on run 6 of 8
- Arithmetic artifact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-asyncoff-extended-runtimeguard-strict-tp4-ctx2048-mbt512-bs256-20260517T140217Z-quality/arithmetic-repeat-n64-r8.json`
- NUL/control counts: `0` / `0`
- Expected repeated token hash:
  `def6899500b2364bc97d561fc5f9cc78aa9fbcd5a0eb032eab1f2c6735d2bbec`
- Divergent run-6 token hash:
  `81659e0f5bc025462926b93246604dd4bb4549ff971ae38139d9f064c07c8213`

## Current Promotion Rule

A MiniMax performance run is not promoted unless the same scheduling/cache class
passes all of these checks before benchmarking:

- raw145 n64 exact
- raw145 n256 exact
- semantic suite
- arithmetic repeat suite
- extended sixpack

The next work item is to isolate whether the failures come from cold cache
compile state, XPU graph replay, async scheduling, or the no-clone allreduce
path itself. Benchmarking resumes only after the quality gate passes again.
