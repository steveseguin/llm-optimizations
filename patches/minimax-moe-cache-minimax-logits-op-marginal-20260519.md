# Focused Patch: Cache MiniMax llm-scaler Logits Callable

Date: 2026-05-19

This is a focused description of the default-off patch tested in
`notes/2026-05-19-minimax-moe-cache-minimax-op-marginal.md`.

The surrounding `moe_wna16.py` file already contains local B70/llm-scaler
changes, so the full repository diff includes unrelated prior work. Apply this
only on top of the existing MiniMax llm-scaler W4A16 logits path.

## File

```text
vllm/model_executor/layers/quantization/moe_wna16.py
```

## Add Module Flag

After `logger = init_logger(__name__)`:

```python
_CACHE_MINIMAX_LOGITS_OP = (
    os.environ.get("VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP", "0") == "1"
)
```

## Cache Callable After Weight Load

After `layer._llm_scaler_moe_u4_decode = True` in
`process_weights_after_loading`:

```python
if _CACHE_MINIMAX_LOGITS_OP and (
    self._llm_scaler_moe_minimax_logits_requested()
    or self._llm_scaler_moe_minimax_logits_ws_requested()
):
    if self._llm_scaler_moe_minimax_logits_ws_requested():
        from custom_esimd_kernels_vllm import (
            moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws,
        )

        layer._llm_scaler_minimax_logits_op = (
            moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws
        )
    else:
        from custom_esimd_kernels_vllm import (
            moe_forward_tiny_cutlass_nmajor_int4_u4_minimax,
        )

        layer._llm_scaler_minimax_logits_op = (
            moe_forward_tiny_cutlass_nmajor_int4_u4_minimax
        )
```

## Use Cached Callable In `apply_monolithic`

Replace the MiniMax logits-path import branch with:

```python
cached_minimax_op = getattr(layer, "_llm_scaler_minimax_logits_op", None)
if cached_minimax_op is not None:
    moe_forward_tiny_cutlass_nmajor_int4_u4_minimax = cached_minimax_op
    if self._llm_scaler_moe_minimax_logits_ws_requested():
        logger.info_once(
            "Using cached llm-scaler XPU INT4 MiniMax logits WS decode path"
        )
    else:
        logger.info_once(
            "Using cached llm-scaler XPU INT4 MiniMax logits decode path"
        )
elif self._llm_scaler_moe_minimax_logits_ws_requested():
    from custom_esimd_kernels_vllm import (
        moe_forward_tiny_cutlass_nmajor_int4_u4_minimax_ws as moe_forward_tiny_cutlass_nmajor_int4_u4_minimax,
    )
    logger.info_once("Using llm-scaler XPU INT4 MiniMax logits WS decode path")
else:
    from custom_esimd_kernels_vllm import (
        moe_forward_tiny_cutlass_nmajor_int4_u4_minimax,
    )
    logger.info_once("Using llm-scaler XPU INT4 MiniMax logits decode path")
```

## Status

Quality passed, but speed delta was too small to promote:

```text
88.549265 output tok/s mean
118.065687 total tok/s mean
```

Keep this as an optional experiment, not as a LocalMaxxing-promoted result.
