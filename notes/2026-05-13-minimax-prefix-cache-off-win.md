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

## No Chunked Prefill Follow-Up

Disabling chunked prefill was tested next because this benchmark uses a single
512-token prompt. The first attempt failed validation:

```text
--no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 512
```

vLLM rejects that combination because `max_num_batched_tokens` must be at least
`max_model_len` when chunked prefill is disabled. The valid equivalent was then
tested with MBT2048:

| Run | Prompt/output | Total tok/s | Output tok/s | KV tokens |
| --- | ---: | ---: | ---: | ---: |
| no prefix, no chunked prefill, MBT2048 | 512/1536 | `92.299677` | `69.224757` | `16128` |

Logs:

- invalid MBT512 warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T173311Z.log`
- MBT2048 warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T173355Z.log`
- MBT2048 measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T173917Z.log`
- AOT: `4be0701399d5053b6e9b2f084fabf3fe9e039f542c8255eda66de3458cafe295`

Decision: do not promote. Disabling chunked prefill forces a larger batching
envelope for this 2048 context, reduces measured KV headroom, and regresses
decode.

## MBT480 Prefix-Off Follow-Up

The prior MBT480 near miss was repeated with prefix caching disabled:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT480, prefix cache off | 512/1536 | `97.623570` | `73.217677` |

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T174334Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T174902Z.log`
- AOT: `9a1ca30e3741b266dc2f4fe430d00426ced0847ccaf31093fce9b0d7635f6349`

Decision: do not promote. Prefix-cache-off also helps MBT480, but the result is
still slightly below MBT512 prefix-off.

## MBT512 Prefix-Off Repeat

The promoted prefix-off cache was repeated with warmup skipped:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT512, prefix cache off repeat | 512/1536 | `97.033748` | `72.775311` |

Logs:

- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T175300Z.log`
- AOT: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

Decision: do not promote. The repeat used a fast init path with no repeated
broadcast waits, but decode was below the promoted 73.232 tok/s run.

## Graph Mode And Async Scheduling

Full graph mode is not a clean next comparison on this XPU build. The XPU
platform hook in:

```text
/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/platforms/xpu.py
```

forces `cudagraph_mode` back to `PIECEWISE` when FlashAttention is active. A
real FULL graph test would require disabling FlashAttention or patching the
platform guard, so it is not currently a useful fast-path experiment.

`--async-scheduling` was tested on the promoted prefix-off MBT512 cache:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| block-size 256, MBT512, prefix cache off, async scheduling | 512/1536 | `97.528481` | `73.146361` |

Logs:

- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T175848Z.log`
- AOT: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

Decision: do not promote. Async scheduling is close but below the simpler
prefix-cache-off best.
