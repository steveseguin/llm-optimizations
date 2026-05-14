# MiniMax Compiled Path Repair Notes

Goal: recover the faster MiniMax M2.7 AutoRound W4A16 TP4 compiled/AOT path
without sacrificing output quality.

## Valid Baseline

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local c51df4300`, Intel XPU
- Stable recipe: TP4, llm-scaler INT4 MoE decode, TRITON_ATTN,
  full-decode-only XPU graph, MiniMax delayed attention allreduce,
  `block-size=256`
- Current promotable result: `61.0808` output tok/s, `81.4411` total tok/s
- LocalMaxxing result: `cmp5e0t6w007ho301nw1qq45h`

The earlier `~73` tok/s compiled/AOT datapoints are not valid performance
claims. They are useful diagnostics only because the raw semantic canary suite
finds corrupt generated output.

## Promotion Rules For This Workstream

A compiled-path result only counts if all of these pass:

- raw semantic canary suite: PASS, arithmetic, and `def add_one`
- no NUL tokens or non-space control output
- nondegenerate generated token distribution
- at least two throughput repeats after quality passes
- LocalMaxxing submission only after the above is true

## Findings

### Reproduced PIECEWISE Corruption

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-piecewise-canary-20260514T125548Z.json`
- Runtime rewrote the config to `VLLM_COMPILE`, `compile_ranges_endpoints=[512]`,
  and `cudagraph_num_of_warmups=1`.
- Output: `192` generated token-id `0` values across three prompts.
- Quality: failed semantic checks and corruption checks.

### Warmup Preservation Did Not Produce A Usable Candidate

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-piecewise-preserve-warmup0-canary-20260514T130517Z.json`
- Patch guard: `VLLM_XPU_PRESERVE_CUDAGRAPH_WARMUPS=1`
- It preserved explicit `cudagraph_num_of_warmups=0`, but the run stalled
  after AOT compile before generation. No benchmark accepted.

### Removing Auto Compile Ranges Fails Shape Coverage

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-piecewise-no-auto-ranges-canary-20260514T131258Z.json`
- Patch guard: `VLLM_XPU_DISABLE_AUTO_COMPILE_RANGES=1`
- Failure: `AssertionError: Shape: 512 out of considered ranges: []`
- Interpretation: vLLM/XPU still needs a 512-token compile/profile shape on
  this path; removing it blindly is not viable.

### Full-Decode Compile Without Inductor Partition Still Corrupts

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-full-decode-no-partition-canary-20260514T131656Z.json`
- Result: same `192` token-id `0` output.
- Interpretation: the corrupt output is not only the inductor graph partition
  pass.

### Finite Trace Places The Failure Before Logits

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-full-decode-no-partition-finite-trace-20260514T132103Z.json`
- Trace:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-full-decode-no-partition-finite-trace-20260514T132103Z.jsonl`
- Real prompt path:
  - `gpu_model_runner.hidden_states_before_logits_index`: shape `[48,3072]`,
    `finite=0`, all NaN
  - selected sample hidden states: all NaN
  - logits after compute: all NaN
  - sampled token: `0`
- Interpretation: token-id `0` is a symptom. The compiled model forward is
  producing NaN hidden states before logits/sampling.

### Eager Fallback For Uncovered Compile Ranges Did Not Repair Quality

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-decode-only-eager-prefill-canary-20260514T132451Z.json`
- Patch guard: `VLLM_XPU_EAGER_FOR_UNCOVERED_COMPILE_RANGES=1`
- It logged fallbacks for shapes `512`, `2`, `48`, `53`, and `57`, but still
  produced `192` token-id `0` outputs.
- Interpretation: the fallback is an eager FX graph path, not a full return to
  the known-good Python/eager model path.

### Runtime Skip For Compiled Prefill Did Not Repair Quality

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-skip-prefill-canary-20260514T133029Z.json`
- Patch guard: `VLLM_XPU_SKIP_COMPILED_PREFILL=1`
- It still produced `192` token-id `0` outputs.
- Interpretation: the corrupt compile/capture state is not avoided simply by
  setting `skip_compiled=True` for padded prefill requests at runtime.

### Disabling AOT Hangs Before A Valid Result

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-aot-canary-20260514T133804Z.json`
- Env: `VLLM_USE_AOT_COMPILE=0`
- Outcome: compiled quickly, then hung around profiling/shared-memory wait.
  Killed. No benchmark accepted.

### Disabling MiniMax Attention Delayed Allreduce Does Not Fix Corruption

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-attn-delay-canary-20260514T134543Z.json`
- Flag: `--no-attention-delay-allreduce`
- Output: `192` generated token-id `0` values.
- Quality: failed semantic checks and corruption checks.
- Interpretation: delayed attention allreduce is not the sole root cause.

### Disabling llm-scaler INT4 MoE Does Not Fix Corruption

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-llm-scaler-moe-canary-20260514T135525Z.json`
- Flag: `--disable-llm-scaler-moe`
- Output: `192` generated token-id `0` values.
- Quality: failed semantic checks and corruption checks.
- Interpretation: the llm-scaler INT4 MoE decode kernel is not the sole root
  cause. The compiled path can still make hidden states non-finite without it.

### PIECEWISE Finite Trace Is Unsafe During Command Graph Capture

- Trace:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-layer-finite-trace-20260514T140153Z.jsonl`
- The trace custom op collected finite warmup/capture records, then crashed in
  command graph capture with:
  `RuntimeError: wait method cannot be used for an event associated with a command graph`.
- Interpretation: the finite trace op is useful for no-cudagraph
  `torch.compile` probes, but it synchronizes via `.item()` and cannot be
  inserted into a captured XPU command graph.

### No-Cudagraph Layer Trace Finds Layer 16 Attention As The First Bad Boundary

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-cudagraph-layer-trace-20260514T141131Z.json`
- Trace:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-cudagraph-layer-trace-20260514T141131Z.jsonl`
- Runtime: `torch.compile` active, `cudagraph_mode=NONE`, tiny one-token
  corruption-tolerant probe.
- Real prompt path:
  - `minimax.layer15.after_moe`: finite
  - `minimax.layer16.input`: finite
  - `minimax.layer16.after_input_norm`: finite
  - `minimax.layer16.after_attn`: all NaN
  - `model.final_hidden`: all NaN
- Interpretation: command graph replay is not the root cause. The model
  forward is already corrupt under `torch.compile` without cudagraph capture,
  and the first real-prompt bad boundary is layer 16 attention.

### No-Cudagraph Attention Trace Finds Q RMSNorm Corruption

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-cudagraph-attn-trace-20260514T141811Z.json`
- Trace:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-no-cudagraph-attn-trace-20260514T141811Z.jsonl`
- Runtime: `torch.compile` active, `cudagraph_mode=NONE`, tiny one-token
  corruption-tolerant probe.
- Real prompt path inside `model.layers.16.self_attn`:
  - `qkv`: finite
  - `k_after_qk_norm`: finite
  - `q_after_qk_norm`: first non-finite tensor
  - `q_after_rope`: more non-finite values
  - `attn_output`: partially non-finite
  - `o_proj_output`: all NaN
- Rank 0 sample for `q_after_qk_norm`, shape `[48,1536]`:
  `finite=73366`, `nan=288`, `posinf=16`, `neginf=58`,
  `min=-64160`, `max=64992`.
- Interpretation: the first actionable compiled-path corruption is MiniMax Q
  RMSNorm under `torch.compile`, not qkv projection, K RMSNorm, RoPE,
  attention, output projection, MoE, logits, or the sampler.

### Splitting Q/K Variance Allreduce Does Not Repair Quality

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-split-qk-var-canary-20260514T142805Z.json`
- Flag: `--split-qk-var-allreduce`
- Output: `192` generated token-id `0` values.
- Interpretation: the packed `[q_var, k_var]` allreduce/chunk sequence is not
  the sole cause of the Q RMSNorm compile corruption.

### Forcing Q/K Norm Through torch.compiler.disable Is Not A Viable Repair

- Command allocated JSON path:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-eager-qk-norm-canary-20260514T143439Z.json`
- Flag: `--eager-qk-norm`
- Startup failed during fullgraph AOT with a graph-break error around the
  `torch.compiler.disable` helper.
- Interpretation: an eager Q/K norm fallback cannot be inserted with
  `torch.compiler.disable` inside this vLLM fullgraph compiled path.

### Decomposed Q/K Norm Expression Does Not Repair Quality

- JSON:
  `/home/steve/bench-results/minimax-m2.7-compile-quality/minimax-compiled-decomposed-qk-norm-canary-20260514T143747Z.json`
- Flag: `--decomposed-qk-norm`
- Output: `192` generated token-id `0` values.
- Interpretation: simple expression decomposition and explicit contiguous
  boundaries are not enough to avoid the Q RMSNorm compile corruption.

## Current Hypotheses

- The bug is in the compiled XPU model-forward path before logits, not in the
  sampler.
- The first real-prompt failure is layer 16 Q RMSNorm under `torch.compile`.
- The bug is not solely caused by command graph capture, delayed attention
  allreduce, inductor graph partition, llm-scaler MoE, or packed Q/K variance
  allreduce.
- The next best repair target is to prevent unsafe Inductor fusion/codegen for
  the Q RMSNorm path while keeping the rest of the fast compiled decode path.

## Next Tests

1. Try narrower Inductor controls around combo/fusion behavior for the compiled
   Q RMSNorm path.
2. If needed, replace Q RMSNorm with an opaque custom op that is safe in
   fullgraph execution, instead of relying on `torch.compiler.disable`.
3. Keep the quality-gated full-decode graph TP4 recipe as the only promoted
   baseline until a compiled candidate passes the semantic suite.
