# MiniMax MoE Full Forward Custom Op Patch Note

Date: 2026-05-19

This patch note records the source change behind `VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP=1`. It was applied to both:

- `/home/steve/src/vllm/vllm/model_executor/models/minimax_m2.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/models/minimax_m2.py`

## Source Changes

1. Added a default-off env guard:

```python
_MINIMAX_M2_MOE_FULL_FORWARD_CUSTOM_OP = (
    os.environ.get("VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP", "0") == "1"
)
_MINIMAX_M2_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS = int(
    os.environ.get("VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS", "4")
)
```

2. Added a MiniMax MoE layer registry and `direct_register_custom_op` wrapper:

```python
_MINIMAX_M2_MOE_LAYER_REGISTRY: dict[str, "MiniMaxM2MoE"] = {}

if TYPE_CHECKING:
    from typing import TypeAlias
    _minimax_layer_name_type: TypeAlias = str | LayerName
else:
    _minimax_layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_minimax_moe_layer_name(layer_name: _minimax_layer_name_type) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    if isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


def _minimax_m2_moe_forward_custom(
    hidden_states: torch.Tensor,
    layer_name: _minimax_layer_name_type,
) -> torch.Tensor:
    layer = _MINIMAX_M2_MOE_LAYER_REGISTRY[
        _resolve_minimax_moe_layer_name(layer_name)
    ]
    return layer._forward_impl_flat(hidden_states)


def _minimax_m2_moe_forward_custom_fake(
    hidden_states: torch.Tensor,
    layer_name: _minimax_layer_name_type,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="minimax_m2_moe_forward",
    op_func=_minimax_m2_moe_forward_custom,
    fake_impl=_minimax_m2_moe_forward_custom_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
```

3. Registered each `MiniMaxM2MoE` instance:

```python
self._encoded_layer_name = LayerName(prefix) if _USE_LAYERNAME else prefix
_MINIMAX_M2_MOE_LAYER_REGISTRY[prefix] = self
```

4. Split the old `MiniMaxM2MoE.forward()` body into `_forward_impl_flat()` and made `forward()` route decode-sized XPU tensors through the custom op:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if (
        _MINIMAX_M2_MOE_FULL_FORWARD_CUSTOM_OP
        and hidden_states.device.type == "xpu"
        and hidden_states.shape[0]
        <= _MINIMAX_M2_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS
        and hidden_states.is_contiguous()
    ):
        final_hidden_states = torch.ops.vllm.minimax_m2_moe_forward(
            hidden_states, self._encoded_layer_name
        )
    else:
        final_hidden_states = self._forward_impl_flat(hidden_states)

    return final_hidden_states.view(num_tokens, hidden_dim)
```

5. Updated the strict quality-gated candidate runner to capture:

```bash
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP
VLLM_MINIMAX_MOE_FULL_FORWARD_CUSTOM_OP_MAX_TOKENS
```

## Validation

- `py_compile` passed for source and venv files.
- Import smoke verified `torch.ops.vllm.minimax_m2_moe_forward` exists.
- Strict quality passed before benchmarking.
- Four long p512/n1536 repeats averaged `89.314195` output tok/s and `119.085594` total tok/s.

## Decision

Keep default-off for controlled benchmarking and promote the env-enabled recipe as the current strict-quality speed high.
