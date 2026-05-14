# MiniMax Full-Decode Graph Triton Result

MiniMax M2.7 AutoRound W4A16 crossed the 60 output tok/s target on four B70s
without changing quantization quality, speculative decoding, expert dropping, or
GPU power limits.

## Result

| Recipe | Prompt/output | Output tok/s | Total tok/s |
| --- | ---: | ---: | ---: |
| TP4, TRITON_ATTN, FULL_DECODE_ONLY graph, run 1 | 512/1536 | 61.761434 | 82.348578 |
| TP4, TRITON_ATTN, FULL_DECODE_ONLY graph, run 2 | 512/1536 | 61.744262 | 82.325683 |
| Mean | 512/1536 | 61.752848 | 82.337130 |

Logs:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T053345Z.log`
- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T053641Z.log`

Quality smoke:

- `/home/steve/bench-results/minimax-m2.7-integrity-gate/graph-full-decode-none-triton-chat-longctx-quality-20260514T054216Z.json`
- `control_char_output=false`, `degenerate_output=false`, `nul_token_count=0`

## Recipe

```bash
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
vllm bench throughput \
  --backend vllm \
  --model /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --tokenizer /mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1536 \
  --random-range-ratio 0 \
  --num-prompts 1 \
  --disable-log-stats \
  --async-engine \
  --block-size 256 \
  --no-enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":0,"compile_sizes":[1]}'
```

## Important Findings

- `CompilationMode.NONE` is required for quality. Inductor/VLLM compiled decode
  paths generated all-NUL token 0 output, even when prefill fell back to eager.
- `FULL_DECODE_ONLY` graph mode is the useful path here. `PIECEWISE` graph mode
  is not compatible with `CompilationMode.NONE`, and vLLM downgrades it to NONE.
- `TRITON_ATTN` avoids the XPU FlashAttention rule that downgrades full graph
  modes to piecewise.
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1` is required for the p512/n1536
  throughput recipe. Without it, the first throughput attempt hung after model
  load with idle GPUs.
- The patch artifact is
  `patches/vllm-xpu-minimax-full-decode-graph-triton-20260514.patch`.

## Next

- Repeat with `p512/n1536` under `max_model_len=4096` and `8192` to see how much
  full-decode graph speed survives larger KV budgets.
- Try `max_num_batched_tokens=1024` and `block_size=128` only after quality smoke.
- Add a stable helper script for this exact recipe so future runs do not omit
  `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE`.
- Follow-up reliability note:
  `notes/2026-05-14-minimax-quality-gate-reliability.md`. The fast graph path
  passes the long-context corruption smoke, but token-exact deterministic greedy
  output is not yet proven and should not be claimed.
