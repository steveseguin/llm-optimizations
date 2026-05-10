# MiniMax DP4+EP Layout Hypothesis, 2026-05-10

## Why This Is Interesting

The MiniMax AutoRound checkpoint is overwhelmingly expert weights:

| Category | Approx size |
| --- | ---: |
| MoE expert weights | `108.712 GiB` |
| embedding + lm head | `2.290 GiB` |
| attention | `1.322 GiB` |
| router gate | `0.091 GiB` |
| norms | `0.001 GiB` |
| total | `112.415 GiB` |

That means a layout with dense weights replicated and experts sharded across
four B70s is roughly:

```text
dense_non_moe + moe_experts / 4 = 3.703 GiB + 108.712 GiB / 4 = 30.881 GiB
```

This is tight but plausible on a 32 GB card.

## Parallelism Implication

The current quality path is TP4. It works, but the clean AOT graph has:

- `62` hidden-state allreduces feeding RMSNorm;
- `63` hidden-state allreduces feeding MoE output handling;
- `62` Q/K variance allreduces feeding Q/K RMS apply.

vLLM's MoE config comments explicitly describe `TP=1, DP=4, EP=true` as four
engine instances with experts split between devices. If this XPU path runs for
a single request, it should keep attention/projection dense weights replicated
and remove many of the TP communication boundaries.

## Next Test

Run only a short smoke first:

```bash
--tensor-parallel-size 1 \
--data-parallel-size 4 \
--enable-expert-parallel \
--all2all-backend allgather_reducescatter \
--gpu-memory-utilization 0.98
```

Use p64/n32 or smaller, then inspect:

- whether it loads without OOM;
- whether `gpu_model_runner` reports per-device memory below the B70 limit;
- whether output throughput beats the TP4 short baseline;
- whether the AOT graph actually removes TP hidden/QK allreduces.

If it works, this becomes the highest-upside path toward `60+` tok/s because it
changes the parallelism boundary instead of shaving individual kernels.
