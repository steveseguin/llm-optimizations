# MiniMax Block-Size 128 Graph Win

Date: 2026-05-13

MiniMax M2.7 AutoRound W4A16 reached a new quality-conservative four-B70 TP4
single-session high:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| attention delay + block-size 128 | 512/1536 | `95.279855` | `71.459891` |

The only delta versus the previous `69.917741` output tok/s promoted recipe is
explicit KV cache block paging:

```text
--block-size 128
```

The run otherwise kept the same quality policy:

- no model weight, quantization, router precision, expert routing, KV dtype,
  sampler, speculative decoding, or power-limit change
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- XPU graph enabled with graph partition, `compile_sizes=[1]`, and cudagraph
  piecewise mode
- llm-scaler XPU INT4 MoE decode path active
- MiniMax Q/K RMS variance allreduces preserved

## Reproduction

```bash
export VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block128-20260513T142239Z
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
export INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1
export WARMUP_INPUT_LEN=512 WARMUP_OUTPUT_LEN=128 WARMUP_NUM_PROMPTS=1
export MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 DTYPE=float16
export FORCE_WARMUP=1 REQUIRE_WARMUP_SUCCESS=1 RUN_TIMEOUT=15m
export EXTRA_ARGS='--async-engine --block-size 128 --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
scripts/bench-vllm-minimax-autoround-xpu-warm-aot.sh
```

Warmup and measured logs:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T142239Z.log`
- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T142806Z.log`

The measured pass directly loaded AOT
`a9b852501268c6a31c53d0f693e89aacaf63891669e825d75b8d59e6249635bb`, reported
17,280 GPU KV tokens and 1.03 GiB available KV memory, then completed in
21.494575 seconds for 1,536 generated tokens.

## Notes

The wrapper also gained a small quoting fix: custom `EXTRA_ARGS` with JSON
compilation config no longer receives an extra closing brace from Bash parameter
expansion. That fix matters for future flag sweeps.
