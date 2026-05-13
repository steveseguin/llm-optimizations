# MiniMax Prefix Cache Off Win

Date: 2026-05-13

MiniMax M2.7 AutoRound W4A16 reached a new four-B70 TP4 single-session high:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT512, prefix cache on | 512/1536 | `97.477615` | `73.108211` |
| block-size 256, MBT512, prefix cache off | 512/1536 | `97.643224` | `73.232418` |

This is a small scheduler/cache win, not a model quality change. The only new
runtime flag versus the prior best is:

```text
--no-enable-prefix-caching
```

Everything quality-sensitive stayed fixed: AutoRound W4A16 model, FP16
activations, KV dtype auto, sampler, routing, TP4, llm-scaler XPU INT4 MoE,
attention delayed-allreduce scheduling, block-size 256, MBT512, and no power
limit change.

## Reproduction

```bash
export VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-20260513T171301Z
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
export INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1
export WARMUP_INPUT_LEN=512 WARMUP_OUTPUT_LEN=128 WARMUP_NUM_PROMPTS=1
export MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 DTYPE=float16
export FORCE_WARMUP=1 REQUIRE_WARMUP_SUCCESS=1 RUN_TIMEOUT=15m
export EXTRA_ARGS='--async-engine --block-size 256 --no-enable-prefix-caching --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
scripts/bench-vllm-minimax-autoround-xpu-warm-aot.sh
```

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T171301Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T172151Z.log`
- AOT: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

The measured pass direct-loaded AOT, reached 17,920 KV tokens, then completed at
`73.232418` output tok/s. The log still showed five shared-memory broadcast
waits before KV setup, so prefix-cache-off does not solve the startup
coordination issue.

Decision: promote and submit. The gain is small, but it is the best
quality-preserving MiniMax 4x B70 number observed so far.
