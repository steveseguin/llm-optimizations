# Grand Plan: MiniMax M2.7 On 4x B70 Beyond 60 Tok/s

Primary goal: improve MiniMax M2.7 AutoRound W4A16 single-session decode on
4x Intel Arc Pro B70 beyond the current quality-gated TP4 baseline without
lowering model quality.

Current best stable baseline:

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: vLLM `0.20.1-local c51df4300`, Intel XPU
- GPUs: 4x Intel Arc Pro B70 32GB
- Recipe: TP4, llm-scaler INT4 MoE decode, TRITON_ATTN, full-decode-only XPU
  graph, MiniMax attention delayed allreduce, `block-size=256`
- Quality-gated fresh repeat: `61.0808` output tok/s, `81.4411` total tok/s
- LocalMaxxing: `cmp5e0t6w007ho301nw1qq45h`
- No power-limit increase.

## Promotion Rules

A new speed result is not promoted unless it satisfies all of these:

- Same model/quantization family, unless explicitly recorded as a separate
  quality tradeoff experiment.
- Long-context corruption gate passes: no NUL tokens, no non-space control
  chars, nondegenerate output.
- Raw semantic canary suite passes.
- At least two throughput repeats for any claimed improvement.
- Notes include decode tok/s, total tok/s, command, logs/JSON paths, and known
  caveats.
- Submit to LocalMaxxing only if the result is valid and useful to share.

## Workstream 1: Quality And Repeatability

Purpose: prevent false wins and make future speed work trustworthy.

Steps:

- Raw canary suite now extends beyond the original `PASS` prompt:
  - final-answer compliance canary
  - arithmetic canary
  - short Python code canary
- Keep exact-token determinism as a diagnostic, not the only pass/fail signal,
  because free-form greedy output can drift even in eager TP4 while remaining
  coherent.
- Add optional finite-logprob/top-logprob capture later if it does not
  destabilize XPU graph execution.

## Workstream 2: Expert Parallelism

Purpose: EP is the biggest plausible quality-preserving speed unlock for this
MoE model.

Known state:

- `--enable-expert-parallel` activates EP ranks on TP4.
- EP-local MoE shape becomes `E=64,N=1536`.
- A first local config for that shape exists in `configs/moe/`.
- Current EP attempts stall before a valid benchmark, so no EP result is
  accepted yet.
- Fresh controls show eager/no-graph EP still stalls, `--no-async-scheduling`
  still stalls before model load, and `--all2all-backend naive` falls back to
  `allgather_reducescatter` on this vLLM build.

Next probes:

- Inspect and, if practical, restore or implement a real non-AG/RS all-to-all
  control path for XPU EP.
- Build a smaller synthetic XCCL/all-to-all repro before launching more full
  MiniMax EP probes.
- If EP reaches generation, run the canary suite before throughput.
- If EP reaches throughput, tune `E=64,N=1536` for decode `M=1`.

## Workstream 3: TP4 Profiling And Fusion

Purpose: improve the stable path if EP remains blocked.

Targets:

- Time per-layer MiniMax MoE dispatch.
- Time TP allreduce and delayed residual allreduce.
- Find whether the current 61 tok/s path is PCIe/oneCCL-bound, kernel-launch
  bound, or MoE GEMM bound.
- Explore quality-preserving fusions only:
  - allreduce plus residual/RMS epilogues
  - MiniMax Q/K RMS scheduling
  - MoE projection epilogue cleanup

## Workstream 4: Graph Reliability

Purpose: graph mode gives the current win but has short-prompt reliability
issues.

Known state:

- Long-context graph quality and throughput can complete.
- Short graph canaries can stall with shared-memory broadcast warnings.

Next probes:

- Compare graph versus eager with the raw canary suite.
- Isolate whether the stall is tied to prompt length, max tokens, async
  scheduling, or communicator graph-capture workaround.
- Avoid claiming graph improvements unless shutdown/runtime hygiene is stable
  enough for repeated runs.

## Workstream 5: Speculative Decode

Purpose: possible later multiplier, but only after the base path is stable.

Rules:

- Do not use speculative decode as a headline until base TP4/EP quality is
  understood.
- Start with ngram speculative decode as a low-risk probe.
- Promote only if canaries pass and accepted-token behavior is recorded.

## Current Immediate Order

1. Keep the expanded raw semantic canary suite as the minimum pre-throughput
   quality screen.
2. Record and push the EP controls from this pass.
3. Inspect vLLM EP all-to-all backend registration and the XPU/XCCL path.
4. If a real EP control path is practical, patch and test it on a synthetic
   repro first.
5. In parallel with EP debugging, move to TP4 timing/profiling because the
   stable 61 tok/s path is still the only promotable path today.
