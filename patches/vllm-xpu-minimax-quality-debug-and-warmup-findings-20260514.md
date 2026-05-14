# MiniMax M2.7 XPU Quality Debug Patch - 2026-05-14

This patch captures the active vLLM/XPU diagnostics used while correcting the
MiniMax M2.7 AutoRound benchmark line:

- finite-value tracing around MiniMax layers, logits, and `gpu_model_runner`
- a safe hidden-state selector experiment
- an opt-in post-forward XPU sync experiment
- an opt-in final-hidden clone/materialization experiment

Key outcome: the earlier compiled/AOT MiniMax runs around 73 output tok/s are
speed-only diagnostics because the quality smoke produced token 0/NUL output.
The current quality-valid fallback is:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1 \
VLLM_XPU_ENABLE_XPU_GRAPH=0 \
vllm bench throughput \
  --backend vllm \
  --model /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1536 \
  --num-prompts 1 \
  --async-engine \
  --block-size 256 \
  --no-enable-prefix-caching \
  --compilation-config '{"mode":0,"cudagraph_num_of_warmups":0}'
```

Repeatability at p512/n1536:

- output tok/s mean: 20.7423
- total tok/s mean: 27.6564
- output tok/s CV: 0.0807%

Important negative findings:

- Default `VLLM_COMPILE` remains invalid even with `cudagraph_num_of_warmups=0`.
- Stock compile remains invalid.
- Disabling llm-scaler MoE did not fix compiled NUL output when tested with a
  fresh cache.
- Cloning/materializing MiniMax final hidden state did not fix compiled NUL
  output.
- `CompilationMode.NONE` is valid only when `cudagraph_num_of_warmups=0`;
  the default one warmup corrupts generation after the first token.

The patch is diagnostic, not a production fix.
