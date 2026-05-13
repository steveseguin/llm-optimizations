# MiniMax M2.7 AutoRound: XPU Graph Breakthrough

## Summary

Re-enabling XPU graph capture for the MiniMax TP4 AutoRound path is the first
run to clear the 60 output tok/s target without changing model quality knobs.
The best screen so far is:

- `67.953543` output tok/s, `90.604724` total tok/s
- prompt/output: `512/1536`
- context length: `2048`
- batch: `1`
- hardware: `4x Intel Arc Pro B70 32GB`
- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- runtime: vLLM `0.20.1`, XPU, TP4, FP16 activations, INC/AutoRound INT4 W4A16,
  llm-scaler INT4 MoE path, FlashAttention, default PCIe topology recognition

This is not a lower-quality path: no expert dropping, no reduced router
precision, no Q/K RMS variance shortcut, no speculative decode, no KV dtype
change, and no weight/quantization change. The win is graph scheduling/capture
around the same compiled TP4 model.

## Launch

```bash
unset CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK
export VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
export VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
export VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-force-p512n256-20260513T101820Z
export DTYPE=float16
export MAX_MODEL_LEN=2048
export MAX_BATCHED_TOKENS=1024
export INPUT_LEN=512
export OUTPUT_LEN=1536
export NUM_PROMPTS=1
export USE_LLM_SCALER_MOE=1
export VLLM_XPU_USE_LLM_SCALER_MOE=1
export XPU_GRAPH=1
export EXTRA_ARGS='--async-engine --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

The graph path requires the local communicator guard recorded in:

```text
patches/vllm-xpu-graph-noop-communicator-capture-20260509.patch
```

## Results

| Run | Prompt | Output | Elapsed | Output tok/s | Total tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| XPU graph p512/n512 | 512 | 512 | `8.352200750` | `61.301209` | `122.602417` |
| XPU graph p512/n1536 | 512 | 1536 | `22.603678052` | `67.953543` | `90.604724` |

Raw result files:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260513T103115Z.json
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T103400Z.json
```

LocalMaxxing:

```text
cmp3y7d720034pc01t8cvx7j3
```

The first submit attempt used `backend: xpu`, but the API backend enum rejected
that value. The accepted submission omits the backend field and records
`XPU/Level Zero` in notes and `engineFlags.extraFlags`.

## Quality Check

I added:

```text
scripts/run-vllm-minimax-quality-check.py
```

The script uses the local MiniMax `chat_template.jinja` and greedy sampling.
The graph/chat run produced coherent MiniMax reasoning text. A strict
graph-vs-eager token comparison is inconclusive because the offline eager/chat
run repeated token `0`/NUL after the first token on this stack. That makes the
eager offline path a poor correctness oracle for this model/runtime pairing.

Saved checks:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/minimax-quality-chat-eager-20260513T1050.json
/home/steve/bench-results/minimax-m2.7-autoround-vllm/minimax-quality-chat-graph-20260513T1054.json
```

Interpretation: the graph result should be reported with a quality caveat:
we have not changed any quality-affecting configuration, and the graph output
is coherent, but the available non-graph offline comparator is not bitwise
usable because it emits NUL repeats.

## Warnings

During graph capture, rank 3 logs an Intel IGC/`ocloc` internal compiler error
for `triton_red_fused__to_copy_mm_t_9`:

```text
IGC: Internal Compiler Error: Floating point exception
```

The run recovers and completes graph capture. This should be kept as a driver
bug to re-test after future Intel compiler/runtime updates.

## Next

- Repeat the accepted p512/n1536 run after any driver/runtime update.
- Run a longer chat-generation sanity pass once we have a better non-graph
  correctness oracle for MiniMax on XPU.
- Continue source-level work on quality-safe collective fusion, but the current
  priority is preserving this graph-capture path because it is the first
  >60 tok/s MiniMax result on the 4x B70 machine.
