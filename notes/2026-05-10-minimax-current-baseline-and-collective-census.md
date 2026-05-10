# MiniMax Current Baseline And Collective Census

Date: 2026-05-10

## Baseline Refresh

Current quality-conservative MiniMax AutoRound recipe:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM/XPU, TP4 on 4x Intel Arc Pro B70 32GB
- dtype: FP16
- MoE path: local llm-scaler raw-u4 decode enabled
- `max_model_len=2048`
- `max_num_batched_tokens=1024`
- `max_num_seqs=1`
- no speculative decoding
- Q/K TP variance allreduce enabled
- no expert dropping
- stock power limits

Fresh p512/n1536 refresh:

| Shape | Total tok/s | Output tok/s | GPU KV tokens | AOT path | Log |
| --- | ---: | ---: | ---: | --- | --- |
| p512/n1536 | `49.557786` | `37.17` | 17,216 | `679011672f...` | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-current-baseline-refresh/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T172305Z.log` |

This is a valid current floor, not a new LocalMaxxing submission. It is close
to the accepted quality-conservative p512/n1536 anchor (`37.552538` output
tok/s / `50.070051` total tok/s, LocalMaxxing `cmozow03v005wlo01q81bnspx`).

Scheduler shape follow-up:

| Shape | `max_num_batched_tokens` | Total tok/s | Output tok/s | GPU KV tokens | Outcome | Log |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| p512/n1536 | 2048 | `45.166742` | `33.88` | 9,344 | Regression | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-mbt-sweep/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T173025Z.log` |
| p512/n1536 | 768 | `43.880692` | `32.91` | 9,280 | Regression | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-mbt-sweep/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T173856Z.log` |

The 2048-token compile range used AOT `f9abbefd...`, took `36.84 s` to
compile, and cut KV headroom almost in half. The 768-token compile range used
isolated cache root
`/mnt/fast-ai/vllm-cache/minimax-m2.7-autoround-mbt768-20260510T173856Z`,
compiled AOT `e08a2767...`, and still produced the same 187 allreduce/wait
boundaries, but the runtime shape was slower than both 1024 and 2048. Keep
`MAX_BATCHED_TOKENS=1024` for this p512/n1536 speed path.

## AOT Collective Census

The loaded cache was:

```text
/home/steve/.cache/vllm/torch_compile_cache/1354e65d1f/rank_0_0/backbone
/home/steve/.cache/vllm/torch_compile_cache/torch_aot_compile/679011672fb322d8dc186c528582d5f2bee43d3132510c02990580dbc9a4ccbf/rank_0_0/model
```

Rank 0 `computation_graph.py` contains the following actual allreduce call
shapes in the backbone graph:

| Allreduce shape | Count | Interpretation |
| --- | ---: | --- |
| `f16[s72, 3072]` | 125 | one vocab embedding allreduce plus 62 output-projection hidden reductions plus 62 MoE hidden reductions |
| `f32[s72, 2]` | 62 | one Q/K RMS variance allreduce per MiniMax layer |

That is 187 TP allreduces per generated-token graph before counting wait/copy
nodes. Comments and stack traces duplicate the string occurrences, but parsing
the actual `torch.ops._c10d_functional.all_reduce(...)` assignments gives the
counts above.

Representative layer-0 pattern:

1. vocab-parallel embedding produces `f16[s72, 3072]`, then allreduces before
   the first RMSNorm;
2. Q/K RMS computes local Q/K variances, concatenates `[q_var, k_var]` into
   `f32[s72, 2]`, allreduces it, then applies Q/K RMS;
3. `o_proj` produces local `f16[s72, 3072]`, allreduces it, then immediately
   feeds residual-add RMSNorm;
4. MoE produces local `f16[s72, 3072]`, allreduces it, then immediately feeds
   the next layer's input residual-add RMSNorm.

## Interpretation

The decode path is dominated by many small synchronization boundaries. The
standalone XCCL microbench showed the tiny Q/K allreduce itself can be fast,
but the compiled graph pays clone, wait, copy, and scheduling barriers for each
collective. The hidden-state allreduces are also immediately followed by
RMSNorm boundaries that are currently decomposed in the graph.

This explains why the recent screens behaved as they did:

- Python custom-op wrapping around allreduce plus fused RMS was too expensive;
- standalone Q/K helper kernels won in isolation but lost in the full AOT graph;
- simply moving output-projection or MoE allreduce boundaries produced small,
  noisy gains but did not reliably beat the p512/n1536 anchor.

## Next Target

The next default-off prototype should target one of these graph boundaries with
a real C++/SYCL or compiler-pass implementation:

- `f16[s72,3072]` allreduce plus residual-add RMSNorm after `o_proj`;
- `f16[s72,3072]` MoE output allreduce plus next-layer input residual-add
  RMSNorm;
- `f32[s72,2]` Q/K variance allreduce fused with Q/K RMS apply, only if the
  fusion preserves the existing INT4 AOT schedule.

The main `60 tok/s` target likely requires reducing the effective cost of the
125 hidden-state reductions, not just replacing the local MoE matvec kernel.
