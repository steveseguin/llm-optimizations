# MiniMax M2.7: Compile Allreduce Custom Op Quality Failure

Date: 2026-05-18

## Candidate

Goal: test whether routing compiled XPU tensor-parallel allreduces through `torch.ops.vllm.all_reduce` improves the promoted 4x B70 MiniMax AutoRound INT4 path without changing model math.

Base recipe:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Runtime: vLLM `0.20.1-local`, XPU TP4, PIECEWISE graph
- Hardware: 4x Intel Arc Pro B70 32GB
- Active promoted baseline to beat: `82.404268` output tok/s, `109.872357` total tok/s
- Promoted env retained: exact MiniMax router logits into llm-scaler INT4 MoE work-sharing, no attention-delay allreduce

Candidate env delta:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
```

The strict runner was also updated to record this env flag in `candidate_env`.

## Result

Rejected before benchmarking.

- Strict status: `quality_failed_raw145_n64`
- Expected raw145 n64 combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Candidate raw145 n64 combined token hash: `fddec0c19f560999e0ab5c4507d694d18e71584e7d4d74342dcf24c45e567678`
- Deterministic across the single-run raw check: yes
- Degenerate output: no
- Benchmark: not run

Relevant warning from the quality log:

```text
vllm::all_reduce ... output of this custom operator ... may not alias any inputs
```

## Interpretation

This is not a safe performance candidate. The output changed under the exact-token gate, and the aliasing warning indicates that the current compiled custom-op wrapper does not satisfy PyTorch's custom-op output contract. Even if a clone could satisfy the contract, it would likely reintroduce the memory traffic this experiment was trying to remove.

## Decision

- Do not promote.
- Do not submit to LocalMaxxing.
- Keep the strict-runner env capture patch because it improves future reproducibility.
- Avoid `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` for the current promoted MiniMax path unless the underlying op is rewritten to be alias-safe and then revalidated from raw hashes upward.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T214429Z-summary.json`
- Raw quality JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T214429Z-quality/raw145-n64-exact.json`
- Raw quality log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-compile-allreduce-custom-op-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T214429Z-quality/raw145-n64-exact.log`
