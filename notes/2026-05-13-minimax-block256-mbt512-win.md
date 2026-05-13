# MiniMax Block-Size 256, MBT512 Win

Date: 2026-05-13

MiniMax M2.7 AutoRound W4A16 reached a new four-B70 TP4 single-session high:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT1024 | 512/1536 | `96.492073` | `72.369055` |
| block-size 256, MBT512 | 512/1536 | `97.477615` | `73.108211` |

This is a scheduler-envelope win, not a quality change. The promoted delta is:

```text
--block-size 256 --max-num-batched-tokens 512
```

All quality-sensitive settings stayed fixed: same AutoRound W4A16 model, FP16
activations, KV dtype auto, sampler, routing, TP4, llm-scaler XPU INT4 MoE
decode path, attention delayed-allreduce scheduling, and no power-limit change.

## Reproduction

```bash
export VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-20260513T152439Z
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
export INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1
export WARMUP_INPUT_LEN=512 WARMUP_OUTPUT_LEN=128 WARMUP_NUM_PROMPTS=1
export MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 DTYPE=float16
export FORCE_WARMUP=1 REQUIRE_WARMUP_SUCCESS=1 RUN_TIMEOUT=15m
export EXTRA_ARGS='--async-engine --block-size 256 --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
scripts/bench-vllm-minimax-autoround-xpu-warm-aot.sh
```

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T152439Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T153001Z.log`
- AOT: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

Measured KV cache size increased to 17,920 tokens versus 17,152 for MBT1024.
The measured pass had repeated worker broadcast waits before generation, but
completed successfully and produced the current best observed score.

## MBT256 Follow-Up

`MAX_BATCHED_TOKENS=256` was tested next with the same block-size 256 recipe.
It further increased measured KV cache size to 18,176 tokens but slowed the
run:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT256 | 512/1536 | `92.672824` | `69.504618` |

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T154117Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T154645Z.log`
- AOT: `9c9be15f0ef17e0b0afa671964895f2e4958296a445630b5a673fabf1e54a412`

Decision: do not promote or submit MBT256. MBT512 remains the best observed
batching envelope for this single-session benchmark.
