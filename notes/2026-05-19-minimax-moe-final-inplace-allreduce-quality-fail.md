# MiniMax MoE Final In-Place Allreduce Rejection

Date: 2026-05-19

## Goal

Test whether the MiniMax MoE final output allreduce can use vLLM's
`torch.ops.vllm.all_reduce_inplace` only at the MoE output boundary. The aim was
to keep the same vLLM/XCCL collective semantics while avoiding an out-of-place
custom-op result tensor for the just-produced local MoE hidden state.

This is different from the promoted tiny Q/K path. The promoted Q/K path only
uses in-place allreduce for the `[tokens, 2]` FP32 variance tensor. This
candidate tried the same no-return custom-op shape on the larger FP16
`[tokens, 3072]` MoE output.

## Patch Shape

Default-off env:

```bash
VLLM_MINIMAX_MOE_FINAL_INPLACE_ALLREDUCE=1
```

When enabled, `MoERunner._maybe_reduce_final_output()` calls:

```python
torch.ops.vllm.all_reduce_inplace(states, group_name=get_tp_group().unique_name)
```

The normal path remains:

```python
states = tensor_model_parallel_all_reduce(states)
```

The strict runner was also updated to record
`VLLM_MINIMAX_MOE_FINAL_INPLACE_ALLREDUCE` in `candidate_env`.

## Result

Rejected before benchmarking.

First gate failed:

- Gate: `raw145-n64-exact`
- Expected combined token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed combined token hash: `a8f3570b8ed4480c708a958eaac3621dd2b473c39415723e1a87c0ce40d73a49`
- Observed text hash: `24dc9247ea9a6699bcc623d8844a8c83b9858696613e849602b7a143e0fbd82d`
- Output was non-degenerate, but not the accepted exact output.

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-final-inplace-ar-quality-screen-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T092307Z-summary.json
```

## Decision

Do not promote and do not submit to LocalMaxxing. This is not a valid speed
candidate because it changes the exact greedy output.

## Lesson

The alias/no-return custom-op route is safe for the tiny FP32 Q/K variance
tensor in the current best recipe, but it should not be generalized to FP16
hidden-state allreduces. A future MoE boundary win needs a genuinely alias-safe
collective/epilogue fusion or a custom op that returns the exact reduced tensor
without changing graph semantics.
