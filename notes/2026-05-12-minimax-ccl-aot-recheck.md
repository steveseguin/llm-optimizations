# MiniMax CCL/AOT Recheck, 2026-05-12

## Goal

After the reboot and follow-up MiniMax work, recheck whether the old `41.130667`
output tok/s p512/n1536 result can be treated as quality-cleared, and record
the oneCCL launcher-env cleanup.

## CCL Local Rank Patch

vLLM's XPU worker had been forcing:

```bash
CCL_PROCESS_LAUNCHER=none
CCL_LOCAL_RANK=<rank>
CCL_LOCAL_SIZE=<local_size>
```

That was originally added for DP/EP experiments, but the faster pre-existing
MiniMax path let oneCCL infer local rank through ATL. I changed this to an
opt-in behavior behind:

```bash
VLLM_XPU_SET_CCL_LOCAL_RANK=1
```

Default TP4 MiniMax runs now show:

```text
could not get local_idx/count from environment variables, trying to get them from ATL
```

instead of using the forced launcher/local-rank env path.

Patch artifact:

`patches/vllm-xpu-worker-ccl-local-rank-optin-20260512.patch`

Both source and installed venv `xpu_worker.py` passed `py_compile`.

## Results

All runs used 4x Arc Pro B70, MiniMax M2.7 AutoRound W4A16, vLLM/XPU TP4,
FP16 activations, llm-scaler u4 MoE enabled, XPU graph disabled, no speculation,
no expert dropping, and no power-limit change.

| Run | Shape | Cache/AOT | KV tokens | Total tok/s | Output tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| CCL ATL smoke | p64/n32 | cold | 9,472 | `18.986` | `6.329` | health check only |
| CCL ATL cold | p512/n1536 | `3a72...` / cold compile | 9,408 | `45.09` | `33.82` | cold artifact |
| CCL ATL warm | p512/n1536 | `8b105...` / AOT `3b096...` | 17,216 | `50.729007` | `38.046755` | valid current floor, small improvement |
| no-wrapper fastpath cold | p512/n1536 | `651f...` / cold compile | 5,184 | `45.59` | `34.19` | negative |
| no-wrapper fastpath warm | p512/n1536 | `9449...` / AOT `243d...` | 17,216 | `49.939924` | `37.454943` | negative; patch removed |

The CCL cleanup recovered the ATL-inferred path and produced a valid warm
`38.046755` output tok/s result, modestly above the earlier
`37.552538` quality-conservative LocalMaxxing reference and above the
`37.751056` attention-delay run. It does not recover the old `41.130667`
speed result.

LocalMaxxing submission:

- id: `cmp27nihp001orm01dataqtfq`
- payload:
  `data/localmaxxing-minimax-m27-autoround-ccl-atl-qkallreduce-p512n1536-20260512.payload.json`
- response:
  `data/localmaxxing-responses/minimax-m27-autoround-ccl-atl-qkallreduce-p512n1536-20260512.response.json`

## AOT Quality Recheck

The old fast archive:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-c158-archive-20260510T070154Z
torch_compile_cache=1d97049441
AOT=c15860ddb8a1077c5ba1a1ae2d0f86552a357eb56772cdbf02828195b5a363ec
```

has mixed signals:

- generated-cache analyzer sees `40` generated allreduce sites, including
  `8` `f32[s72,2]` sites classified as Q/K variance;
- the top-level `rank_0_0/backbone/computation_graph.py` for cache key
  `1d97049441` shows MiniMax Q/K RMS from old `linear_attn.py:93-107` with
  local `q_var`/`k_var` calculation and no visible
  `tensor_model_parallel_all_reduce(qk_var)`.

The current warm graph:

```text
torch_compile_cache=8b105d58a9
AOT=3b0962f0af06261f1111081e2233aff35e7dd7899eb46ceced593d08e3d91b71
```

shows the expected top-level Q/K variance communication:

```text
linear_attn.py:115 qk_var = torch.cat([q_var, k_var], dim=-1)
all_reduce_1: "f32[s72, 2]" = torch.ops._c10d_functional.all_reduce(...)
linear_attn.py:117 qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
linear_attn.py:118 q_var, k_var = qk_var.chunk(2, dim=-1)
```

Decision: keep the `41.130667` p512/n1536 run as a useful scheduling clue, not
as the quality-cleared baseline. The current quality-cleared non-spec baseline
is now the `38.046755` warm CCL-ATL result unless a later run beats it while
retaining the top-level Q/K allreduce signature.

## Next Work

The next useful path is not another Python wrapper or deleted helper branch.
The remaining target is a real XPU/Level Zero-side fusion around either:

- Q/K variance allreduce plus RMS apply, preserving the cross-rank variance;
- hidden-state allreduce plus adjacent residual/RMS or MoE epilogue work;
- MiniMax candidate router repair fused into the existing working
  `moe_int4_ops` extension, not the standalone SYCL module that crashed during
  device-image registration.

Structured data:

`data/minimax-m27-ccl-aot-recheck-20260512.json`
