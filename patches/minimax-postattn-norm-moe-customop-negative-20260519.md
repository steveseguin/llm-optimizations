# MiniMax Post-Attention Norm Plus MoE Custom Op Patch Note

Date: 2026-05-19

## Changed Files

- `/home/steve/src/vllm/vllm/model_executor/models/minimax_m2.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/models/minimax_m2.py`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`

## Default-Off Flags

```bash
VLLM_MINIMAX_POST_ATTN_NORM_MOE_CUSTOM_OP=1
VLLM_MINIMAX_POST_ATTN_NORM_MOE_CUSTOM_OP_MAX_TOKENS=4
```

## Implementation Shape

- Added a MiniMax decoder-layer registry keyed by layer prefix.
- Registered `torch.ops.vllm.minimax_m2_post_attn_norm_moe`.
- The custom op calls the existing Python/PyTorch implementation:
  1. `self.post_attention_layernorm(hidden_states, residual)`
  2. `self.block_sparse_moe(hidden_states)`
- The guard requires XPU, contiguous `hidden_states` and `residual`, no delayed
  attention allreduce path, no AR+RMS XPU helper path, and token count less than
  or equal to `VLLM_MINIMAX_POST_ATTN_NORM_MOE_CUSTOM_OP_MAX_TOKENS`.
- The strict runner now records both new env vars in `candidate_env`.

## Result

The patch is exact-quality but slower on the current high-speed stack:

- `89.007143` output tok/s mean
- `118.676191` total tok/s mean
- current promoted high remains `89.314195` output tok/s

Decision: keep the code default-off as a reproducibility artifact, but do not
enable it in the promoted MiniMax recipe.
