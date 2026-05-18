# MiniMax M2.7 Logits-WS No-Attention-Delay Small Win

Date: 2026-05-18

## Result

`VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0` was retested on top of the current exact MiniMax router-logits-to-work-sharing path:

- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- default XPU FlashAttention v2
- PIECEWISE XPU graph
- p512/n1536, ctx2048, batch 1, TP4

This is a different context than the older attention-delay wins. With the newer logits-WS MoE path, the delayed attention allreduce is no longer helpful; turning it off gives a small but repeatable gain.

Four clean long runs:

- Run 1: `82.852197` output tok/s, `110.469596` total tok/s
- Run 2: `82.681455` output tok/s, `110.241939` total tok/s
- Run 3: `82.400523` output tok/s, `109.867364` total tok/s
- Run 4: `81.682897` output tok/s, `108.910529` total tok/s
- Mean: `82.404268` output tok/s, `109.872357` total tok/s

This is `+0.79%` output tok/s over the previous strict logits-WS promoted result of `81.758267` output tok/s, and `+2.24%` over the earlier work-sharing baseline of `80.602755` output tok/s. The fourth run drifted lower, so this should be treated as a conservative small promotion rather than a large new tier.

One isolated fresh-cache confirmation attempt was aborted by the startup stall guard during graph compile and produced no benchmark JSON. It is not counted as a performance run.

## Quality

The full strict quality gate passed before benchmarking:

- raw145 n64 exact hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite hash: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic hash: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

No speculative decoding, expert dropping, router approximation, quantization change, or power-limit change was used.

## Command

```bash
LABEL=logits-ws-no-attn-delay-screen-20260518 \
ATTENTION_BACKEND=default \
VLLM_XPU_USE_LLM_SCALER_MOE=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1 \
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1 \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1 \
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2 \
VLLM_XPU_ENABLE_XPU_GRAPH=1 \
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1 \
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
BENCH_REPEATS=2 \
QUALITY_TIMEOUT=35m \
BENCH_TIMEOUT=30m \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

Warm-cache confirmation repeats used the same runtime flags and cache root:

```bash
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-strict-logits-ws-no-attn-delay-screen-20260518 \
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
OUTDIR=/home/steve/bench-results/minimax-m2.7-strict-candidates \
TP=4 MAX_MODEL_LEN=2048 MAX_BATCHED_TOKENS=512 MAX_NUM_SEQS=1 \
INPUT_LEN=512 OUTPUT_LEN=1536 NUM_PROMPTS=1 DTYPE=float16 \
XPU_GRAPH=1 RUN_TIMEOUT=30m RUN_TIMEOUT_KILL_AFTER=30s SHM_STALL_MAX_WARNINGS=6 \
EXTRA_ARGS='--async-engine --block-size 256 --no-enable-prefix-caching --compilation-config {"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}' \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-screen-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T172328Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-no-attn-delay-screen-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T172328Z-quality`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T173919Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T174208Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T175053Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T175341Z.json`

## Decision

Promote as the current strict MiniMax baseline because it is quality-preserving and repeatable across four clean long runs.

- LocalMaxxing id: `cmpbifcx3013bmn01747cxix8`

The gain is small. The next meaningful path is still reducing decode-critical GPU/CPU/framework boundaries and collective overhead, not weakening quality checks.
