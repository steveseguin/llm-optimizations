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

## RMSNorm Provider Priority Control

Tested:

```bash
--ir-op-priority '{"rms_norm":["xpu_kernels","native"]}'
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
```

Result:

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/rms-xpu-priority/compiled-piecewise-raw145-rms-xpu-priority-clean-weight-ctx2048-n64-20260515T120232Z.json`
- AOT hash:
  `1adc6533d493aed52ae552ea450e8ca00e84179d617c46903552857b0a7b395e`
- AOT graph: `1496` allreduce calls, `1496` wait-tensor calls,
  `0` allreduce/RMS/MoE boundary lines.
- Quality: passed; `64` generated tokens, `28` distinct token ids, `0` NUL,
  `0` non-space control characters.
- Three p512/n1536 repeats: `65.4250`, `65.9902`, `66.3541` output tok/s.
- Mean: `65.9231` output tok/s, `87.8974` total tok/s.

Decision: neutral diagnostic. This restored the boundary count to the old
fast-graph shape, but did not deliver a meaningful throughput gain over the
accepted `65.7525` tok/s baseline. Do not submit to LocalMaxxing as a new win.

## Custom RMS Op Control

Added harness support for:

```bash
--compilation-custom-ops 'none,+rms_norm'
```

Tested with the clean Q/K weight guard enabled.

Result:

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/custom-rms-op/compiled-piecewise-raw145-custom-rms-op-clean-weight-ctx2048-n64-20260515T122132Z.json`
- AOT hash:
  `e970952d6f89295d3966efc4ee9a87f45890fccc42b51034a54240246322c323`
- AOT graph: `1496` allreduce calls, `1496` wait-tensor calls,
  `32` RMS+INT4 source lines, `24` compiled INT4/RMS kernels, and `0`
  allreduce/RMS/MoE boundary lines.
- Quality: passed with the same token and text hashes as the accepted
  clean-weight canary; `64` generated tokens, `28` distinct token ids, `0`
  NUL, `0` non-space control characters.
- Three p512/n1536 repeats: `65.1308`, `65.7245`, `66.0467` output tok/s.
- Mean: `65.6340` output tok/s, `87.5120` total tok/s.

Decision: neutral diagnostic. The fused RMS/INT4 markers alone do not explain
the old invalid `~73` tok/s result. This candidate is quality-valid but is
slightly below the accepted baseline. Do not submit to LocalMaxxing.

## CCL P2P/USM And Topology Controls

Record:
`data/minimax-m27-ccl-usm-and-topology-controls-20260515.json`

USM control:

```bash
CCL_TOPO_P2P_ACCESS=0
```

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/ccl-usm/compiled-piecewise-raw145-ccl-usm-clean-weight-ctx2048-n64-20260515T123841Z.json`
- Quality: passed with the same raw-prompt token/text hashes as the accepted
  clean-weight canary; `64` generated tokens, `28` distinct token ids, `0`
  NUL, `0` non-space control characters.
- Three p512/n1536 repeats: `60.5218`, `59.7768`, `60.0204` output tok/s.
- Mean: `60.1063` output tok/s, `80.1417` total tok/s.

Decision: reject USM mode for this TP4 MiniMax decode path. It is quality-safe
but about `8.6%` slower than the accepted P2P baseline.

Topology-recognition override:

```bash
CCL_TOPO_P2P_ACCESS=1
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
```

- Quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/ccl-topology-assume-xelink/compiled-piecewise-raw145-ccl-topology-assume-xelink-clean-weight-ctx2048-n64-20260515T125120Z.json`
- Quality: passed with the same raw-prompt token/text hashes as the accepted
  clean-weight canary; `64` generated tokens, `28` distinct token ids, `0`
  NUL, `0` non-space control characters.
- Three p512/n1536 repeats: `66.1840`, `65.9685`, `66.2300` output tok/s.
- Mean: `66.1275` output tok/s, `88.1700` total tok/s.

Decision: useful CCL boundary condition, but not a meaningful new public result.
The mean is only about `0.6%` above the accepted `65.7525` tok/s baseline and
below the accepted run maximum of `66.6589` tok/s. Do not submit to
LocalMaxxing unless later repeats show a larger, stable gap.

## Serving TTFT Split And Long-Prefill Control

Record:
`data/minimax-m27-serve-and-prefill-controls-20260515.json`

Serving p512/n1536 for the accepted clean-weight piecewise recipe:

- Result JSON:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-aot-cleanweight/vllm-minimax-m27-autoround-serve-tp4-p512n1536-np1-20260515T131006Z.json`
- Output throughput: `65.9048` tok/s
- Total throughput: `87.8730` tok/s
- Mean TTFT: `605.50` ms
- Mean TPOT/ITL: `14.7886` ms/token

Interpretation: for the single p512/n1536 request shape, TTFT is not the main
ceiling. The steady decode interval is about `14.79` ms/token, so the next
headline speed work still needs to reduce decode-step cost.

Serving p512/n1536 with
`CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`:

- Result JSON:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-aot-cleanweight-ccltopo/vllm-minimax-m27-autoround-serve-tp4-p512n1536-np1-20260515T131505Z.json`
- Output throughput: `65.4255` tok/s
- Total throughput: `87.2339` tok/s
- Mean TTFT: `765.84` ms
- Mean TPOT/ITL: `14.7954` ms/token

Decision: reject as a serving/decode improvement. The direct throughput repeats
for this oneCCL topology override were slightly positive, but serving ITL is
effectively unchanged and TTFT/output throughput are worse. Keep default
oneCCL topology recognition for serving.

Long-prefill p4096/n512 with `max_model_len=8192` and `MBT=1024` was then
screened because prefill is still a separate optimization target:

- Serving JSON:
  `/home/steve/bench-results/minimax-m2.7-serve-metrics-prefill-cleanweight/vllm-minimax-m27-autoround-serve-tp4-p4096n512-np1-20260515T131811Z.json`
- Direct throughput JSON:
  `/home/steve/bench-results/minimax-m2.7-prefill-cleanweight-throughput/vllm-minimax-m27-autoround-tp4-p4096n512-20260515T132414Z.json`
- Long-context quality JSON:
  `/home/steve/bench-results/minimax-m2.7-quality-gated/prefill-longctx-8192-mbt1024-cleanweight-20260515T132810Z.json`

Timing:

- Serving p4096/n512: `24.5689` output tok/s, `221.1204` total tok/s,
  `8164.85` ms TTFT, `24.8015` ms mean ITL.
- Direct throughput p4096/n512: `39.6848` computed output tok/s,
  `357.1635` total tok/s.

Quality result:

- Long-context canary passed: `false`
- Generated tokens: `64`
- Distinct token ids: `1`
- NUL tokens: `64`
- Non-space control characters: `64`

Decision: reject `max_model_len=8192` / `MBT=1024` for this clean-weight
piecewise path. The random serving output was all NUL characters and the
long-context quality canary reproduced the same NUL-token failure. These timing
numbers are retained only as failure diagnostics and must not be submitted to
LocalMaxxing or compared as valid performance.

## Long-Context Graph Controls

Record:
`data/minimax-m27-long-context-graph-controls-20260515.json`

Follow-up controls after the `8192`/`MBT1024` NUL-token failure:

- Full eager, `max_model_len=8192`, `MBT=1024`: passed the long-context
  canary with `16` distinct generated tokens, `0` NUL tokens, and `0`
  non-space control characters.
- Graph mode with `--skip-compiled-prefill`: hung after cached graph/AOT load
  with repeated shared-memory broadcast waits. No JSON result was produced.
- Graph mode with `MBT=512`: compiled the `512` range, then hung with repeated
  shared-memory broadcast waits. No JSON result was produced.
- Decode-graph plus eager fallback for uncovered prefill shapes:
  `VLLM_XPU_DISABLE_AUTO_COMPILE_RANGES=1` and
  `VLLM_XPU_EAGER_FOR_UNCOVERED_COMPILE_RANGES=1` reached generation but
  produced `16/16` NUL tokens. Reject for quality.
- `--cudagraph-mode none` with torch.compile still compiled the `1024` range
  and then hung. This points beyond XPU graph replay alone; the compiled
  piecewise prefill path is suspect.
- `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1` did not fix the NUL-token output.
- `--skip-compiled-prefill` plus `--cudagraph-mode none` also hung after
  compiling the `1024` range, so original-Python prefill is not yet a usable
  workaround inside this graph/compile harness.

Conclusion: long-context correctness is currently only confirmed under full
eager execution. The compiled/piecewise path for larger prefill shapes remains
unsafe. Do not publish long-context compiled-graph numbers until the first
generated-token NUL corruption is fixed and a quality canary passes.

Finite tracing narrowed the failure:

- Graph path trace:
  `/home/steve/bench-results/minimax-m2.7-finite-trace/longctx-graph-token0-20260515T142246Z.trace.jsonl`
- Eager reference trace:
  `/home/steve/bench-results/minimax-m2.7-finite-trace/longctx-eager-reference-20260515T142535Z.trace.jsonl`
- Graph generated token id `0`; eager generated token id `758`.
- Graph `gpu_model_runner.hidden_states_before_logits_index` for the first
  `1024 x 3072` prefill chunk had `0` finite values and `3145728` NaNs on all
  ranks.
- Graph `gpu_model_runner.logits_after_compute` for that chunk had `0` finite
  values and `200064` NaNs.
- The second `382 x 3072` chunk was also all NaN before logits.
- Eager, same prompt and model settings, kept those tensors finite:
  first chunk min/max approximately `-61.41` / `64.56`, logits min/max
  approximately `-13.44` / `32.09`, with `0` NaNs.

This means token id `0` is a symptom of NaN hidden states/logits from the
compiled long-prefill forward path, not a sampler-only issue.

Additional serving control:

- `VLLM_XPU_SET_CCL_LOCAL_RANK=1` on p512/n1536 serving produced `64.7703`
  output tok/s, `86.3604` total tok/s, `840.98` ms TTFT, and `14.9011` ms
  mean ITL.
- Decision: reject for speed. It is below the accepted `65.7525` tok/s mean
  and does not reduce decode ITL.

Harness note: a late current-best repeatability canary using the existing 2k
cache hung after cached AOT load before prompt processing. Since serving with
the same accepted cache still completed, treat this as harness/runtime
instability after repeated graph experiments, not as a quality failure of the
published baseline.

## Deferred XPU Output-Copy Screen

Record:
`data/minimax-m27-deferred-output-copy-20260515.json`

Patch:
`patches/vllm-xpu-deferred-output-copy-inference-mode-20260515.patch`

The default-off `VLLM_XPU_DEFER_ASYNC_OUTPUT_COPY=1` path first failed in the
async output thread:

```text
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

Wrapping `finish_deferred_xpu_output_copy()` in `torch.inference_mode()` fixed
the crash. The completed p512/n1536 runs were slower than a same-session
baseline repeat:

| Candidate | Output tok/s | Total tok/s | Decision |
| --- | ---: | ---: | --- |
| same-session baseline | `66.1183` | `88.1577` | keep |
| deferred copy, global XPU sync | `62.4365` | `83.2486` | reject |
| deferred copy, no global XPU sync | `62.0221` | `82.6961` | reject |

Decision: keep `VLLM_XPU_DEFER_ASYNC_OUTPUT_COPY` unset for MiniMax TP4 decode.
This was not submitted to LocalMaxxing because it is a negative result.

## Skip-Compiled Prefill Profile Patch

Record:
`data/minimax-m27-skip-compiled-prefill-profile-run-20260515.json`

Patch:
`patches/vllm-xpu-skip-compiled-prefill-profile-run-20260515.patch`

Root cause found in the first patched attempt:

- `VLLM_XPU_SKIP_COMPILED_PREFILL=1` covered live prefill in one runner path,
  but profile/dummy prefill still entered the compiled MiniMax backbone.
- With `VLLM_XPU_DISABLE_AUTO_COMPILE_RANGES=1`, startup failed during
  memory profiling with `Shape: 1024 out of considered ranges: []`.
- The fix passes `skip_compiled=True` through `set_forward_context` for
  profile/dummy runs when the token count is greater than one. I also patched
  the sync runner path so the env control is consistent.

Validated controls:

- Long-context, `max_model_len=8192`, `MBT=1024`,
  `--disable-auto-compile-ranges`, `--skip-compiled-prefill`,
  `--cudagraph-mode none`: passed. It generated `48` tokens across the fixed
  canary prompts, `34` distinct token ids, `0` NUL tokens, and `0` non-space
  control characters.
- Same long-context settings with piecewise graph capture: rejected. It hung
  after model load with repeated shared-memory broadcast waits while the
  workers remained CPU-bound.
- Same long-context settings with `FULL_DECODE_ONLY` and the default
  FlashAttention backend: rejected. XPU/SYCL graph capture fails because
  `sycl_ext_oneapi_work_group_scratch_memory` is not available for the SYCL
  Graph extension.
- Same long-context settings with `FULL_DECODE_ONLY` and `TRITON_ATTN`:
  rejected for quality. It reached generation, but produced `45/48` NUL
  tokens and only `2` distinct token ids.
- Current accepted p512/n1536 piecewise recipe after the patch: non-regressing.
  One repeat produced `66.0956` output tok/s and `88.1275` total tok/s.
- Current accepted raw 145-token quality canary after the patch: passed and
  matched the expected token hash
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`.

Decision: keep this as a correctness/workaround patch, not a new public speed
result. It gives us a quality-correct long-context control when graph capture
is disabled, and it does not disturb the accepted 2k-context decode baseline.
Do not submit to LocalMaxxing because there is no new speed win.

## Next Work

1. Continue to treat `65.7525` output tok/s as the published quality-valid
   baseline; treat `66.1275` with topology override as a local repeatability
   datapoint, not yet a new tier.
2. Keep P2P enabled. USM mode is rejected for this path.
3. Move beyond graph-shape counters: the provider-priority and custom-RMS
   controls show that nicer AOT markers do not automatically improve decode.
4. Target actual runtime costs next: communication scheduling, allreduce
   placement, TTFT/prefill, or a real fused allreduce/RMS/MoE epilogue path.
5. Run the raw quality canary before any speed benchmark.
6. Promote only repeatable improvements that clearly exceed the current
   noise band, ideally by at least `3%` on mean output tok/s while preserving
   token/text quality hashes.
7. Keep long-context work behind quality gates. The `8192` compiled/piecewise
   prefill path currently corrupts into token id `0` or hangs; debug it with
   finite tracing and source-level shape guards before spending more benchmark
   time on long-context speed.
8. For long-context speed, do not use the currently tested graph-capture
   variants. The no-graph prefill workaround is correct but slow; piecewise
   capture hangs; FlashAttention full-decode capture fails at SYCL Graph
   startup; Triton full-decode capture corrupts into NUL tokens.
