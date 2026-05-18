# MiniMax MoE Work-Sharing Flash PIECEWISE Strict Win

Date: 2026-05-18

## Summary

The llm-scaler ESIMD work-sharing INT4 MoE decode kernel was retested inside
the current strict MiniMax FlashAttention/PIECEWISE recipe by enabling
`VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`. This time it is a real win.

The run passed the full strict quality harness before benchmarking: raw145 n64
and n256 exact token hashes, semantic suite, 16-repeat arithmetic canary, and
extended sixpack. Two p512/n1536 repeats averaged `80.602755` output tok/s and
`107.470340` total tok/s. That is `+15.136343%` versus the promoted
FlashAttention/PIECEWISE strict baseline mean of `70.006353` output tok/s.

No model weights, quantization, router precision, expert routing, KV dtype,
sampler, speculative decoding, or power setting was changed. This is a
quality-gated speed result.

LocalMaxxing accepted the mean result as `cmpasdq5v007nmn019elaut3s`.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Base model: `MiniMaxAI/MiniMax-M2.7`
- Engine: vLLM `0.20.1-local`, XPU TP4
- Hardware: 4x Intel Arc Pro B70 32GB
- Quantization: AutoRound INT4 W4A16 / INC
- Dtype: `float16`
- Attention backend: default XPU FlashAttention v2
- Shape: p512, n1536, ctx2048, batch 1
- Block size: 256
- Max batched tokens: 512
- Prefix cache: disabled
- Temperature: greedy / 0
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
- `CCL_TOPO_P2P_ACCESS=1`
- `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`
- `ZE_AFFINITY_MASK=0,1,2,3`

Benchmark extra args:

```bash
--async-engine \
--block-size 256 \
--no-enable-prefix-caching \
--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}'
```

## Quality

Strict quality summary:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

The arithmetic repeat gate matters because it has caught previous unsafe
full-logits/local-argmax and scheduling variants. This candidate passed it
before any speed result was considered.

## Results

Two strict-gated p512/n1536 repeats:

| repeat | elapsed s | output tok/s | total tok/s |
| --- | ---: | ---: | ---: |
| 1 | 19.211613 | 79.951642 | 106.602189 |
| 2 | 18.903716 | 81.253867 | 108.338490 |
| mean | - | 80.602755 | 107.470340 |

Sample standard deviation for output tok/s: `0.920812`; min/max spread:
`1.615609%` of the mean.

Baseline comparison:

- promoted strict baseline mean output tok/s: `70.006353`
- MoE work-sharing mean output tok/s: `80.602755`
- delta: `+15.136343%`

The vLLM benchmark JSON reports total tok/s for prompt plus decode but does not
separately expose prefill tok/s for this run. The submitted total tok/s is the
repeat mean of those prompt-plus-decode measurements.

## Implementation Detail

The installed vLLM path selects the work-sharing kernel only for tiny decode
batches:

- vLLM gate:
  `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/moe_wna16.py`
- llm-scaler Python wrapper:
  `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/ops.py`
- llm-scaler SYCL kernel:
  `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`

The active `moe_wna16.py` code imports
`moe_forward_tiny_cutlass_nmajor_int4_u4_ws` when
`VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`; otherwise it uses the normal
`moe_forward_tiny_cutlass_nmajor_int4_u4` path. Prompt/prefill still uses
vLLM fused experts.

The AOT hash stayed at
`03f6a28c070656d44eab4c581bc8dc5295ed123e7c0150c7f596ea24012406b0` because
the kernel choice happens inside the custom MoE op path rather than as a new
Inductor graph pattern.

Earlier work-sharing screens were negative in older MiniMax runtime recipes.
The current result shows the same kernel can become positive when combined with
the stricter FlashAttention/PIECEWISE graph schedule, block-size 256,
max-batched-tokens 512, no prefix cache, and the current quality-gated runtime.

## Decision

Promote this as the current honest MiniMax M2.7 AutoRound TP4 speed result on
4x B70: `80.602755` output tok/s mean at p512/n1536 after strict quality
gating.

Do not combine this with unsafe logits/router/argmax shortcuts unless those
future variants pass the same strict quality harness. The next speed work
should keep this recipe as the baseline and focus on hidden-state collective
boundaries, MoE/projection epilogue fusion, and prefill efficiency without
changing model quality.

## Artifacts

- Strict summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T052722Z-summary.json`
- Strict quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-strict-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T052722Z-quality`
- Benchmark JSON 1:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T054336Z.json`
- Benchmark JSON 2:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T054627Z.json`
- LocalMaxxing payload:
  `data/localmaxxing-minimax-m27-autoround-moe-ws-flash-piecewise-strict-p512n1536-20260518.payload.json`
- LocalMaxxing response:
  `data/localmaxxing-responses/minimax-m27-autoround-moe-ws-flash-piecewise-strict-p512n1536-20260518.response.json`
