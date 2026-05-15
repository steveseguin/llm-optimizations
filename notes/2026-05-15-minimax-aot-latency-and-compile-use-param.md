# MiniMax AOT Latency And Compile-Use-Param Control

Goal: continue improving MiniMax M2.7 AutoRound W4A16 on 4x B70 without
accepting speed-only results that corrupt output.

## Current Accepted Baseline

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: vLLM `0.20.1-local`, XPU, TP4, piecewise/AOT compile
- Required correctness flag:
  `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- Current promoted LocalMaxxing result: `cmp6a5c1o00mpo3011hg8ncyp`
- Three p512/n1536 repeats: `64.6223`, `66.6589`, `65.9762` output tok/s
- Mean: `65.7525` output tok/s, `87.6699` total tok/s
- Quality gates: raw-prompt 64-token and 256-token canaries passed with no
  NUL tokens, no non-space control characters, and nondegenerate output.

This remains the best quality-valid MiniMax result so far. The older `~73`
tok/s AOT family remains speed-only diagnostic data because it produced token
id `0` / NUL output on the raw prompt canary.

## Server Latency Probe

Result:

- JSON:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-aot/vllm-minimax-m27-autoround-serve-tp4-p512n256-np3-20260515T113800Z.json`
- Server log:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-aot/vllm-minimax-m27-autoround-serve-server-tp4-p512n256-np3-20260515T113800Z.log`
- Benchmark log:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-aot/vllm-minimax-m27-autoround-serve-bench-tp4-p512n256-np3-20260515T113800Z.log`

Metrics:

- Requests: `3`
- Prompt tokens: `1536`
- Output tokens: `768`
- Output throughput: `65.0856` tok/s
- Total throughput: `195.2569` tok/s
- Mean TTFT: `4603.8070` ms
- Mean TPOT/ITL: `13.5665` ms

Interpretation: the accepted recipe's steady decode interval is about
`13.57` ms/token, which corresponds to about `73.7` tok/s once the prompt and
first-token phase are out of the way. The p512/n256 serving number stays near
`65` tok/s because TTFT is still about `4.6` seconds. Prefill/TTFT is therefore
a separate optimization track, but it should not be mixed with headline
long-output decode claims.

## AOT Graph Comparison

Compared the accepted clean-weight AOT graph against the older invalid fast
AOT graph.

Accepted clean-weight graph:

- AOT hash: `13b321b34e7fd5459622f6904ca57e7598c7831539ca3104d19c1bcbbdee374d`
- Actual allreduce calls: `1496`
- Actual wait-tensor calls: `1496`
- `rms_norm + int4_gemm_w4a16` source lines: `32`
- Compiled INT4 RMS kernels: `24`
- Aten RMS/MoE boundary lines after allreduce: `16`
- Compiled allreduce/RMS/MoE boundary kernels: `12`

Old invalid fast graph:

- AOT hash: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`
- Actual allreduce calls: `1496`
- Actual wait-tensor calls: `1496`
- `rms_norm + int4_gemm_w4a16` source lines: `32`
- Compiled INT4 RMS kernels: `24`
- Aten RMS/MoE boundary lines after allreduce: `0`
- Compiled allreduce/RMS/MoE boundary kernels: `0`

Conclusion: the remaining speed gap is not explained by extra collective call
count. The useful target is restoring the faster vLLM IR RMSNorm graph shape
around the attention-output/allreduce/MoE boundary while preserving the Q/K
clean-weight corruption fix.

## Negative Control: Compile Uses Live Param

Added a default-off source toggle:

```bash
VLLM_MINIMAX_QK_NORM_COMPILE_USE_PARAM=1
```

The experiment returns `norm.weight` during Dynamo compile or XPU stream
capture, while keeping the clean-weight guard available outside compiled
regions. Rationale: check whether the clean XPU clone was what pushed the graph
away from the old faster shape.

Result:

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/compile-use-param/compiled-piecewise-raw145-compile-use-param-ctx2048-n64-20260515T114454Z.json`
- Throughput JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/compile-use-param-throughput/vllm-minimax-m27-autoround-tp4-p512n1536-20260515T115030Z.json`
- Quality: passed; `64` generated tokens, `28` distinct token ids, `0` NUL,
  `0` non-space control characters.
- Throughput: `64.4976` output tok/s, `85.9968` total tok/s.
- AOT hash: unchanged at
  `13b321b34e7fd5459622f6904ca57e7598c7831539ca3104d19c1bcbbdee374d`.

Decision: reject this as a speed route. It is quality-safe in this canary, but
it does not improve speed or graph shape. Do not submit this control to
LocalMaxxing.

## Next Work

1. Inspect why the valid clean-weight graph has Aten RMS/MoE boundary kernels
   where the old invalid graph stayed in `vllm_ir.rms_norm`.
2. Try a minimal graph-shape repair that does not change model math or router
   behavior.
3. Run the raw quality canary before any speed benchmark.
4. Promote only repeatable improvements over `65.7525` output tok/s.
