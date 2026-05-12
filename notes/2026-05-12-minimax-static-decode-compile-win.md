# MiniMax Static Decode Compile Win, 2026-05-12

## Result

MiniMax M2.7 AutoRound W4A16 reached a new repeatable four-B70 TP4 result with
vLLM/XPU by combining the previous graph-partition win with an explicit static
decode compile size:

```bash
--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1]}'
```

The quality path is unchanged: no speculative decode, no expert dropping, no
power-limit change, FP16 activations, AutoRound INT4 W4A16 weights, and the
MiniMax Q/K RMS variance allreduce remains present in the generated graph.

| Run | Prompt | Output | AOT | KV tokens | Output tok/s | Total tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| warm screen | 512 | 512 | `3e2cefa1` | 16,832 | `45.430028` | `90.860056` |
| validation | 512 | 1536 | `3e2cefa1` | 16,832 | `47.376673` | `63.168898` |
| validation repeat | 512 | 1536 | `3e2cefa1` | 16,064 | `47.586110` | `63.448146` |

Previous quality-cleared long-run best was `40.209882` output tok/s and
`53.613176` total tok/s with graph partition only, so this is about an 18.4%
decode improvement on the same benchmark shape.

## Command Shape

```bash
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
VENV=/home/steve/.venvs/vllm-xpu \
HF_HOME=/mnt/fast-ai/llm-cache/hf \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-inductor-partition-compile1-p512n512-20260512T122329Z \
LLM_SCALER_KERNELS=/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python \
USE_LLM_SCALER_MOE=1 \
XPU_GRAPH=0 DTYPE=float16 TP=4 \
MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 \
INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1 \
EXTRA_ARGS='--compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1]}' \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

## Graph Check

The top-level cached graph still shows:

- `q_var = q.pow(2).mean(...)`
- `k_var = k.pow(2).mean(...)`
- `qk_var = torch.cat([q_var, k_var], dim=-1)`
- `_c10d_functional.all_reduce(...)` on `f32[s72, 2]`
- divide by TP world size and `chunk(2)` before Q/K RMS apply

Analyzer summary for rank 0 included `187` allreduces:

- `125` hidden-state `f16[s72,3072]` allreduces
- `62` Q/K variance `f32[s72,2]` allreduces

## Important Caveat

The first cold compile emitted Intel `ocloc` / IGC internal compiler errors for
one static Triton reduction kernel:

```text
IGC: Internal Compiler Error: Floating point exception
triton_red_fused__to_copy_mm_t_9
xnumel=256, r0_numel=3072, XBLOCK=4, R0_BLOCK=128
```

The run still completed after compilation fallback/retry behavior, and warm
reloads used the `3e2cefa1...` AOT graph successfully. This is a driver/compiler
bug lead to preserve, not a blocker for using the warm cache.

## Negative Screens In This Pass

- `CCL_ZE_IPC_EXCHANGE=pidfd` with graph partition: `39.71` output tok/s,
  below the `40.21` graph-partition best.
- `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` with graph partition: first run
  `40.62`, repeat `39.98`, so not promoted.
- attention delayed-allreduce plus graph partition: compiled, then hung after
  request start; killed manually.
- `--stream-interval 16`: hung during early worker/oneCCL setup; killed
  manually.
- `--no-async-scheduling`: compiled a new graph but regressed to `29.45` output
  tok/s at p512/n512.

## Decision

Promote `use_inductor_graph_partition=true` plus `compile_sizes=[1]` as the new
MiniMax M2.7 AutoRound TP4 four-B70 raw-speed recipe, pending future work on
true collective/RMS fusion and driver compiler stability.

LocalMaxxing accepted the repeated p512/n1536 result as
`cmp2mf1zw007wrm01op7aimhk`.
