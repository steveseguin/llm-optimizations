# MiniMax M2.7 AutoRound FP8 KV Screen, 2026-05-10

## Goal

Check whether vLLM FP8 KV cache improves the four-B70 MiniMax AutoRound path
enough to justify a separate quality-tradeoff track. This is not a
quality-preserving baseline until output quality is validated, but it could
reduce KV memory pressure or expose longer-context headroom.

## Command

```bash
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-kvfp8-screen-p512n512 \
TP=4 PORT=18088 INPUT_LEN=512 OUTPUT_LEN=512 NUM_PROMPTS=1 \
MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 \
EXTRA_SERVER_ARGS='--kv-cache-dtype fp8' \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-serve-xpu.sh
```

## Result

The model loaded and initialized, but the benchmark completed zero requests.
This should not be submitted to LocalMaxxing as a performance result.

Observed details:

- Model load: `28.11 GiB` per TP worker, `73.79 s`.
- AOT compile/warmup: `20.99 s`.
- FP8 KV capacity: `34,496` tokens at `max_model_len=2048`.
- Prior comparable FP16/BF16 KV capacity was about `17,216` tokens, so FP8 KV
  roughly doubled reported KV capacity.
- The first benchmark request reached the API server, then the engine timed out
  inside `sample_tokens`.
- vLLM reported repeated shared-memory broadcast stalls before the fatal
  timeout.

Key error:

```text
TimeoutError: RPC call to sample_tokens timed out.
EngineDeadError: EngineCore encountered an issue.
```

Benchmark JSON:

```json
{
  "completed": 0,
  "total_input_tokens": 0,
  "total_output_tokens": 0,
  "output_throughput": 0.0,
  "total_token_throughput": 0.0,
  "mean_ttft_ms": 0.0,
  "mean_tpot_ms": 0.0
}
```

## Assessment

FP8 KV is a useful capacity signal but is not usable as a MiniMax TP4 speed
path on the current XPU/vLLM stack. It likely needs an XPU attention/KV
debugging pass before it can be benchmarked meaningfully.

Keep this as a future longer-context track, not as part of the current
quality-preserving 60 tok/s target. The near-term optimization effort should
stay on MiniMax TP4 collective/graph fusion with the standard KV path.

