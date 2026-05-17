# MiniMax Packed-Gather And Pair-Reuse Patch Note

Date: 2026-05-17

Status: rejected as performance promotions. Both branches are default-off.

## Runtime Flags

Packed gather:

```bash
export VLLM_XPU_LOCAL_ARGMAX_PACKED_GATHER=1
```

Pair reuse:

```bash
export VLLM_XPU_LOCAL_ARGMAX_PAIR_REUSE=1
```

Both are intended to run only with the strict MiniMax local-argmax baseline:

```bash
export VLLM_XPU_LOCAL_ARGMAX_DECODE=1
export VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_XPU_USE_LLM_SCALER_MOE=1
export CCL_TOPO_P2P_ACCESS=1
```

## Patch Surface

Runtime files patched locally:

- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`

Harness files patched:

- `scripts/bench-vllm-minimax-autoround-xpu.sh`
- `scripts/inspect-vllm-runtime.py`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`

## Packed-Gather Branch

Inserted before the default float32 pair gather in `get_top_tokens`:

```python
if (
    os.environ.get("VLLM_XPU_LOCAL_ARGMAX_PACKED_GATHER", "0") == "1"
    and logits.device.type == "xpu"
):
    with timed_region("logits.local_argmax_packed_gather_key"):
        value_bits = local_max_vals.float().view(torch.int32).to(torch.int64)
        value_bits = value_bits & 0xFFFFFFFF
        ordered_bits = torch.where(
            (value_bits & 0x80000000) != 0,
            (~value_bits) & 0xFFFFFFFF,
            value_bits ^ 0x80000000,
        )
        signed_value_key = ordered_bits - 0x80000000
        tie_key = (0xFFFFFFFF - global_indices.to(torch.int64)) & 0xFFFFFFFF
        packed = ((signed_value_key << 32) | tie_key).contiguous()
    with timed_region("logits.local_argmax_packed_gather_collective"):
        gathered = torch.empty(
            (tp_size,) + tuple(packed.shape),
            dtype=packed.dtype,
            device=packed.device,
        )
        dist.all_gather_into_tensor(
            gathered, packed, group=get_tp_group().device_group
        )
    with timed_region("logits.local_argmax_packed_gather_reduce"):
        top_packed = gathered.max(dim=0).values
        top_tokens = (0xFFFFFFFF - (top_packed & 0xFFFFFFFF)).to(torch.int64)
    return top_tokens
```

Result: full strict quality passed, but mean output was only `57.670263` tok/s
versus `61.404035` tok/s baseline.

## Pair-Reuse Branch

Inserted before the default float32 pair gather in `get_top_tokens`:

```python
if (
    os.environ.get("VLLM_XPU_LOCAL_ARGMAX_PAIR_REUSE", "0") == "1"
    and logits.device.type == "xpu"
):
    with timed_region("logits.local_argmax_pair_reuse_fill"):
        pair_shape = tuple(local_max_vals.shape) + (2,)
        cache_key = (str(logits.device), torch.float32, pair_shape)
        cached = getattr(self, "_xpu_local_argmax_pair_reuse_cache", None)
        if cached is None or cached[0] != cache_key:
            local_pair = torch.empty(
                pair_shape, dtype=torch.float32, device=logits.device
            )
            gathered = torch.empty(
                (tp_size,) + pair_shape,
                dtype=torch.float32,
                device=logits.device,
            )
            self._xpu_local_argmax_pair_reuse_cache = (
                cache_key,
                local_pair,
                gathered,
            )
        else:
            local_pair = cached[1]
            gathered = cached[2]
        local_pair[..., 0].copy_(local_max_vals.float())
        local_pair[..., 1].copy_(global_indices.float())
    with timed_region("logits.local_argmax_pair_reuse_collective"):
        dist.all_gather_into_tensor(
            gathered, local_pair, group=get_tp_group().device_group
        )
    with timed_region("logits.local_argmax_pair_reuse_reduce"):
        gathered_view = gathered.movedim(0, 1)
        max_rank_idx = gathered_view[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered_view[:, :, 1].gather(dim=-1, index=max_rank_idx)
        top_tokens = top_tokens.squeeze(-1).to(torch.int64)
    return top_tokens
```

Result: raw145 n64 exact token hash passed, but the first p512/n1536 throughput
run was only `45.498382` output tok/s, so the full strict gate was skipped.

## Harness Updates

`scripts/bench-vllm-minimax-autoround-xpu.sh` now records these env flags in
logs so benchmark JSONs can be tied back to the exact candidate:

```bash
vllm_xpu_local_argmax_pair_reuse=${VLLM_XPU_LOCAL_ARGMAX_PAIR_REUSE:-}
vllm_xpu_local_argmax_packed_gather=${VLLM_XPU_LOCAL_ARGMAX_PACKED_GATHER:-}
```

`scripts/inspect-vllm-runtime.py` now detects the pair-reuse, packed-gather,
packed-allreduce, and allreduce local-argmax markers in runtime diagnostics.

## Reproduction Artifacts

- Data: `data/minimax-m27-packed-gather-and-pair-reuse-negative-20260517.json`
- Note: `notes/2026-05-17-minimax-packed-gather-and-pair-reuse-negative.md`
