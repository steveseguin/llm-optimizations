# MiniMax Distributed-Residual Allreduce Screen

Date: 2026-05-13

This screen tested a source-level variant of the current MiniMax M2.7
AutoRound best recipe. The idea was to distribute the residual add across TP
ranks before the hidden-state allreduce:

```python
hidden_states = hidden_states + residual * (1.0 / tp_size)
all_reduce(hidden_states)
```

That is algebraically equivalent to adding the full residual on rank 0 before
the allreduce, assuming the residual tensor is identical on all TP ranks. The
goal was to see whether this shape would compile or schedule better on XPU graph
without sacrificing quality.

## Setup

Base recipe:

- TP4, FP16 activations
- XPU graph with graph partitioning
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_MINIMAX_M2_DIST_RESIDUAL_ALLREDUCE=1`
- `--block-size 256`
- `MAX_BATCHED_TOKENS=512`
- `MAX_NUM_SEQS=1`
- `--no-enable-prefix-caching`
- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-distresid-block256-mbt512-noprefix-20260514T001936Z`
- AOT hash: `d2ff2c916a54632fb1b730da99bb20edf7c3983f0242653dfc2099e311597bc5`

The patch is default-off and was applied to both the live venv copy and the
source checkout:

- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/models/minimax_m2.py`
- `/home/steve/src/vllm/vllm/model_executor/models/minimax_m2.py`

Repro patch: `patches/vllm-minimax-dist-residual-allreduce-20260513.patch`

## Result

| Run | Prompt/output | Total tok/s | Output tok/s | Decision |
| --- | ---: | ---: | ---: | --- |
| Distributed residual allreduce | 512/1536 | `96.897489` | `72.673116` | negative |

Current validated best for the same prompt/output size is `73.306312` output
tok/s submitted, with a repeat mean of `73.244155` output tok/s. This variant
is below both.

## AOT Collective Check

The generated graph still contains the same collective shape:

- actual allreduce calls: `1496`
- actual wait-tensor calls: `1496`
- allreduce/wait pairs within 7 lines: `1496`
- wait gap: `{ "2": 1496 }`
- categories:
  - `embedding_hidden`: `8`
  - `qk_rms_variance`: `496`
  - `attention_o_proj_hidden`: `496`
  - `moe_hidden`: `496`

So this did not solve the true bottleneck. It changed the arithmetic placement
around hidden-state allreduce but left the immediate allreduce/wait boundaries
intact.

## Decision

Do not promote or submit to LocalMaxxing. Keep the env-gated patch recorded as a
negative source experiment, but leave `VLLM_MINIMAX_M2_DIST_RESIDUAL_ALLREDUCE`
unset for current-best runs.

The next useful work is lower than this Python-level placement change: either
remove or fuse the XPU collective wait boundary in the backend, or add a real
device-side collective plus adjacent op fusion path for RMS/residual/projection
and MoE epilogues.
