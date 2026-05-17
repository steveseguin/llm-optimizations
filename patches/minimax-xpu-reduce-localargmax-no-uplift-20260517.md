# MiniMax Local-Argmax XPU Reduce Patch Notes

Date: 2026-05-17

Status: default-off, quality-safe, not a speed win.

## Patch Surface

Added a standalone extension:

```text
experiments/minimax_pair_argmax_xpu/
```

The extension exposes two functions:

- `pair_argmax(values, tokens, group_name, world_size)`
- `reduce_flat_pairs(gathered, world_size)`

Only `reduce_flat_pairs` is safe enough to wire into vLLM. The full helper's
functional c10d gather path failed a corrected `rank_wins` oracle in standalone
testing.

The vLLM local-argmax path was tested with this default-off branch after the
existing pair all-gather:

```python
if (
    os.environ.get("VLLM_XPU_LOCAL_ARGMAX_XPU_REDUCE", "0") == "1"
    and gathered.device.type == "xpu"
):
    with timed_region("logits.local_argmax_xpu_reduce"):
        reducer = getattr(self, "_xpu_local_argmax_reduce_flat_pairs", None)
        if reducer is None:
            import minimax_pair_argmax_xpu

            reducer = minimax_pair_argmax_xpu.reduce_flat_pairs
            self._xpu_local_argmax_reduce_flat_pairs = reducer
        top_tokens = reducer(gathered.contiguous(), tp_size)
    return top_tokens
```

Runtime guard support was added to:

- `scripts/inspect-vllm-runtime.py`
- `scripts/bench-vllm-minimax-autoround-xpu.sh`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`

The local vLLM source and installed venv copy were both patched for the screen:

- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`
- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`

## Build

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export MAX_JOBS=1
export MINIMAX_PAIR_ARGMAX_XPU_SYCL_TARGETS=spir64_gen,spir64
export MINIMAX_PAIR_ARGMAX_XPU_SYCL_DEVICE=bmg
python -m pip install --no-build-isolation -e /home/steve/llm-optimizations-publish/experiments/minimax_pair_argmax_xpu
```

Do not publish or rely on the generated `.so` artifact; rebuild it locally.

## Test Command

```bash
LABEL=minimaxlogits-localargmax-xpu-reduce-screen \
BENCH_REPEATS=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=8 \
RUN_EXTENDED_QUALITY=0 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-strict-minimaxlogits-localargmax-xpu-reduce-screen-20260517 \
VLLM_RUNTIME_REQUIRE_MARKERS=VLLM_XPU_LOCAL_ARGMAX_XPU_REDUCE \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1 \
VLLM_XPU_LOCAL_ARGMAX_DECODE=1 \
VLLM_XPU_LOCAL_ARGMAX_ASSUME_SAFE=1 \
VLLM_XPU_LOCAL_ARGMAX_XPU_REDUCE=1 \
VLLM_BENCH_TEMPERATURE=0 \
CCL_TOPO_P2P_ACCESS=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZE_AFFINITY_MASK=0,1,2,3 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Result

Strict quality passed, but p512/n1536 output throughput was `60.071619` tok/s
versus the promoted strict baseline `61.404035` tok/s.

Decision: keep disabled and do not submit to LocalMaxxing.
