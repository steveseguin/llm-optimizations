# MiniMax MoE Final In-Place Allreduce Patch

Status: rejected by quality gate on 2026-05-19.

## Code Touches

- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`

## Runtime Guard

```bash
VLLM_MINIMAX_MOE_FINAL_INPLACE_ALLREDUCE=1
```

The guard is default-off.

## Intended Change

Add a MoE-only final allreduce path:

```python
if (
    _minimax_moe_final_inplace_allreduce_enabled()
    and self.moe_config.tp_size > 1
    and hasattr(torch.ops, "vllm")
    and hasattr(torch.ops.vllm, "all_reduce_inplace")
):
    torch.ops.vllm.all_reduce_inplace(
        states, group_name=get_tp_group().unique_name
    )
else:
    states = tensor_model_parallel_all_reduce(states)
```

The strict runner was updated to capture
`VLLM_MINIMAX_MOE_FINAL_INPLACE_ALLREDUCE` in the summary JSON.

## Validation Result

The first raw145 exact-token quality gate failed:

- Expected token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed token hash: `a8f3570b8ed4480c708a958eaac3621dd2b473c39415723e1a87c0ce40d73a49`

No throughput benchmark was run.

## Reproduction

Use the current strict harness and set:

```bash
export VLLM_MINIMAX_MOE_FINAL_INPLACE_ALLREDUCE=1
export BENCH_REPEATS=0
export RUN_REPEAT_ARITHMETIC_QUALITY=0
export RUN_EXTENDED_QUALITY=0
export LABEL=minimax-moe-final-inplace-ar-quality-screen-20260519
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

Keep this patch as a negative reference only. Do not use the env for accepted
MiniMax benchmarks.
