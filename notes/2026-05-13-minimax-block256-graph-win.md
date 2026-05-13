# MiniMax Block-Size 256 Graph Win

Date: 2026-05-13

MiniMax M2.7 AutoRound W4A16 reached a new four-B70 TP4 single-session high:

| Run | Prompt/output | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: |
| attention delay + block-size 128 | 512/1536 | `95.279855` | `71.459891` |
| attention delay + block-size 256 | 512/1536 | `96.492073` | `72.369055` |
| attention delay + block-size 384 | 512/1536 | `94.087055` | `70.565291` |
| attention delay + block-size 512 | 512/1536 | `93.436845` | `70.077633` |

This keeps the same quality policy as the previous run. The only promoted delta
is:

```text
--block-size 256
```

That changes KV cache paging, not model weights, routing, KV precision, sampler,
speculative decoding, or power policy.

## Reproduction

```bash
export VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-20260513T143351Z
export VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
export INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1
export WARMUP_INPUT_LEN=512 WARMUP_OUTPUT_LEN=128 WARMUP_NUM_PROMPTS=1
export MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=1024 MAX_NUM_SEQS=1 DTYPE=float16
export FORCE_WARMUP=1 REQUIRE_WARMUP_SUCCESS=1 RUN_TIMEOUT=15m
export EXTRA_ARGS='--async-engine --block-size 256 --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
scripts/bench-vllm-minimax-autoround-xpu-warm-aot.sh
```

Logs:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T143351Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T144250Z.log`

Measured pass direct-loaded AOT
`7a930122cb9f1c8a67e6ae8a3543bfbd71795ee76045bbbcb9c1063aaaf242c1`, reported
17,152 GPU KV tokens and 1.03 GiB available KV memory, then completed in
21.224541 seconds for 1,536 generated tokens.

## Notes

The cold warmup emitted repeated rank-3 `ocloc` / IGC internal compiler errors
for Triton reduction kernels but recovered and saved a valid AOT artifact. The
measured pass had long worker broadcast waits after direct AOT load, then
generated successfully. Treat block-size 256 as current best but not yet the
end of the block-size sweep; block-size 512 or a repeat run should be tested
before assuming the curve peaks here.

Block-size 512 was tested next and regressed. It remained above 70 output tok/s
but was materially slower than block-size 256:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T145357Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T145923Z.log`
- AOT: `ca103e551c598f64689aabc5412febc685eecd21b5d7eb175a6156d73b5a0591`
- KV tokens: 16,896

Current decision: promote block-size 256, do not submit block-size 512 to
LocalMaxxing.

Block-size 384 was also tested and landed between 256 and 512, but still below
the 256 high:

- warmup: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260513T150413Z.log`
- measured: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T151512Z.log`
- AOT: `29d37262ad244dd19cfe44ce9b41d48cc6b9ca897e9b7bd2a44e4c25126f8c84`
- measured KV tokens: 15,360

Current block-size decision: `256` is the best observed page size for this
quality-preserving p512/n1536 recipe.
