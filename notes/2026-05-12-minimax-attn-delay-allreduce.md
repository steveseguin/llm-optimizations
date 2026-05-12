# MiniMax Attention Delayed Allreduce, 2026-05-12

## Goal

Test whether the MiniMax M2.7 TP4 decode path can improve by moving the
attention residual add before the attention output allreduce. This targets one
of the hidden-state collective boundaries without removing any TP
communication.

## Patch

Default-off env:

```bash
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
```

The local vLLM patch constructs MiniMax attention `o_proj` with
`reduce_results=false` when the flag is set. The decoder then adds the residual
on TP rank 0 before `tensor_model_parallel_all_reduce()` and applies
post-attention RMSNorm without the residual argument.

I also screened `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1`, but MoE-delay and
attention+MoE-delay were negative.

## Results

All runs used:

- 4x Intel Arc Pro B70 32GB, stock power limits
- MiniMax M2.7 AutoRound W4A16 from
  `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- vLLM `0.20.1-local`, XPU/Level Zero, FP16 activations
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- B70 `E=256,N=384,dtype=int4_w4a16` MoE config
- XPU graph disabled, no speculative decode, no expert dropping

Short p64/n32 warmed screen:

| Run | Total tok/s | Output tok/s | Outcome |
| --- | ---: | ---: | --- |
| no-flag control | `77.015741` | `25.671914` | baseline |
| attention delay | `79.617726` | `26.539242` | small positive |
| MoE delay | `74.463709` | `24.821236` | negative |
| attention+MoE delay | `71.970752` | `23.990251` | negative |

Promoted p512/n1536 attention-delay validation:

| Run | KV tokens | Total tok/s | Output tok/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| cold compile | `9,408` | `45.329862` | `33.997396` | cold AOT artifact |
| warm repeat 1 | `17,280` | `49.848540` | `37.386405` | below reference |
| warm repeat 2 | `17,280` | `50.334741` | `37.751056` | promoted |

The previous quality-conservative LocalMaxxing reference was `37.552538`
output tok/s and `50.070051` total tok/s on the same p512/n1536 shape. This is
a small improvement, not a major breakthrough.

## Graph Check

The promoted AOT graph still contains `187` allreduces:

- `62` Q/K RMS variance allreduces: `f32[s72, 2] -> qk_rms_apply`
- `62` hidden-state attention/residual RMS boundaries:
  `f16[s72, 3072] -> rms_norm`
- `63` hidden-state MoE/final boundaries: `f16[s72, 3072] -> moe`

Quality note: this path keeps all Q/K TP variance communication and does not
drop experts. It does change floating-point accumulation order for the
attention residual path, so I am treating it as quality-conservative but not
bitwise-identical.

## Native Router Attempt

A separate native MiniMax candidate-router repair op was attempted in
`custom_esimd_kernels_vllm`. A standalone SYCL extension compiled, but loading
the `.so` segfaulted during SYCL device-image registration
(`ProgramManager::addImage` / `__sycl_register_lib`). Linking the simple generic
kernel into `moe_int4_ops` produced the same load crash. I restored the active
runtime to the known-loadable llm-scaler u4-logits binary.

Next native-router attempt should reuse the existing llm-scaler ESIMD/top-k
patterns or isolate a JIT-only build path, not the simple generic SYCL module.

Detailed data: `data/minimax-m27-attn-delay-allreduce-20260512.json`.
