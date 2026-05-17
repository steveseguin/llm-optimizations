# MiniMax Direct-Gather Reuse Patch Note

Date: 2026-05-17

This patch was tested as a default-off local-argmax candidate for MiniMax M2.7
AutoRound INT4 on 4x Intel Arc Pro B70.

## Status

Rejected as a performance promotion. It passed the strict quality gate, but its
mean decode rate was `61.289497` tok/s versus the promoted strict baseline of
`61.404035` tok/s.

No LocalMaxxing submission was made for this candidate because it did not
improve the promoted result.

## Runtime Flag

Enable only for reproduction:

```bash
export VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER_REUSE=1
export VLLM_RUNTIME_REQUIRE_MARKERS=logits.local_argmax_pair_direct_gather_reuse
```

The candidate also assumes the promoted strict MiniMax environment:

```bash
export VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1
export VLLM_XPU_LOCAL_ARGMAX_DECODE=1
export VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export CCL_TOPO_P2P_ACCESS=1
```

## Patch Surface

Files patched locally:

- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`

The branch is inserted before the existing
`VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER` branch in the local-argmax token
selection path:

```python
if (
    os.environ.get("VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER_REUSE", "0") == "1"
    and local_pair.device.type == "xpu"
):
    with timed_region("logits.local_argmax_pair_direct_gather_reuse"):
        input_pair = local_pair.contiguous()
        gathered_shape = (tp_size,) + tuple(input_pair.shape)
        cache_key = (str(input_pair.device), input_pair.dtype, gathered_shape)
        cached = getattr(self, "_xpu_local_argmax_direct_gather_reuse_cache", None)
        if cached is None or cached[0] != cache_key:
            gathered = torch.empty(
                gathered_shape,
                dtype=input_pair.dtype,
                device=input_pair.device,
            )
            self._xpu_local_argmax_direct_gather_reuse_cache = (cache_key, gathered)
        else:
            gathered = cached[1]
        dist.all_gather_into_tensor(
            gathered, input_pair, group=get_tp_group().device_group
        )
    with timed_region("logits.local_argmax_direct_gather_reuse_reduce"):
        gathered_view = gathered.movedim(0, 1)
        max_rank_idx = gathered_view[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered_view[:, :, 1].gather(dim=-1, index=max_rank_idx)
        top_tokens = top_tokens.squeeze(-1).to(torch.int64)
    return top_tokens
```

## Result Artifacts

- Result data:
  `data/minimax-m27-direct-gather-reuse-no-improvement-20260517.json`
- Result note:
  `notes/2026-05-17-minimax-direct-gather-reuse-no-improvement.md`
- Quality/bench summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-direct-gather-reuse-tightquality-strict-tp4-ctx2048-mbt512-bs256-20260517T160531Z-summary.json`

## Lesson

Standalone XCCL `all_gather_into_tensor` for the tiny pair payload measures
about `0.098` to `0.117` ms on rank 0, but the vLLM runtime timing bucket around
the per-token local-argmax gather path is still much larger. The next bottleneck
to investigate is framework, stream, graph replay, or CPU handoff overhead
around the collective rather than raw XCCL payload transfer.
