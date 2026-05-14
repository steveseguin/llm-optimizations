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
- Use the synthetic XCCL/AG-RS repro to validate an XPU-safe padded equal-size
  `all_gatherv` path before launching more full MiniMax EP probes.
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
2. Treat the EP path as blocked until async sampled-token transfer and worker
   synchronization are understood; do not spend full-model benchmark time on
   EP until a focused repro changes that.
3. Keep the XPU uneven `all_gatherv` padded equal-size patch as a real candidate
   fix because the synthetic repro proves raw uneven XCCL gather can hang.
4. Move the main effort to repairing the faster TP4 compiled/AOT recipe. It
   previously produced about `73` output tok/s, but is speed-only diagnostic
   data until the raw semantic canary suite passes.
5. Once a compiled candidate passes quality, run at least two throughput
   repeats and submit only then.

## Active Blockers As Of 2026-05-14

- EP is not yet a valid route to a headline number. The furthest run reaches
  decode only with diagnostic sync-copy enabled, then hangs in Level Zero
  XPU-to-CPU transfer.
- The stable promotable MiniMax number is still the full-decode graph TP4 path:
  about `61.08` output tok/s, quality-gated and submitted.
- The compiled/AOT TP4 path is still useful as a diagnostic target because it
  previously demonstrated `~73` output tok/s without changing model
  quantization, but repeated repair attempts still corrupt output and it does
  not count.
- Compiled-path finite tracing shows the failure happens before logits:
  real-prompt hidden states become all-NaN before sampler selection, which then
  degenerates to token-id `0`.
- No-cudagraph compiled traces narrow the first real-prompt corruption to layer
  16 attention, specifically Q RMSNorm. The qkv projection and K RMSNorm are
  still finite at that boundary.
- Disabling MiniMax delayed attention allreduce, disabling llm-scaler INT4 MoE,
  splitting the Q/K variance allreduce, decomposing the Q/K norm expression,
  and replacing local Q/K RMS work with return-value or allocating SYCL helper
  custom ops do not fix the compiled-path corruption.
- Immediate priority shifts back to valid-quality paths: profile the stable
  TP4 graph recipe, repair EP communication with focused repros, and only use
  the compiled path for cheap one-token diagnostics until it stops producing
  NUL output.
- Raising `max_num_batched_tokens` to `1024` is now a rejected stability
  candidate for the current TP4 graph recipe. It passed the semantic quality
  screen but stalled before throughput and left the XPU runtime unhealthy,
  requiring reboot-level recovery. Keep the promoted recipe at
  `max_num_batched_tokens=512` unless a separate graph/shared-memory fix lands.
- A post-reboot repeat of the promoted recipe reproduced `61.0167` output
  tok/s mean with the quality gate passing.
- The older faster piecewise graph/AOT recipe still fails the expanded raw
  semantic canary with all generated tokens equal to `0`; it remains invalid
  even if its throughput is higher.
- The direct Q/K RMS XPU helper remains off. A valid-path screen first hit a
  transient oneCCL/OFI startup failure, then a retry stalled before quality JSON
  with shared-memory broadcast waits. The quality-gated runner now saves a
  quality log and applies `QUALITY_TIMEOUT` so future quality-stage hangs are
  bounded.
- Disabling vLLM graph memory estimation with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` preserves quality on the canary
  but is rejected for the promoted recipe: the throughput stage stalls after
  model load with repeated shared-memory broadcast waits and produces no JSON.
- A strict two-run token-hash determinism check on the promoted graph recipe
  failed with the first token difference at generated token index `24`, while
  both runs remained printable, nondegenerate, and NUL-free. Current wording
  should stay precise: semantic quality/corruption gates pass, but token-exact
  deterministic greedy decoding is not yet proven.
- `max_model_len=4096` is currently rejected for the promoted TP4 graph recipe.
  The quality stage reports `17,408` KV cache tokens and `4.25x` theoretical
  concurrency, then stalls in `sample_tokens` with shared-memory broadcast waits
  and no quality JSON.
- A short eager/no-cudagraph 4096-context smoke passes with no NUL/control
  output, so the 4096-context blocker is likely in full-decode XPU graph or
  shared-memory scheduling, not the model, prompt, or KV allocation alone.
- A non-sync timing probe on the valid path only sees uncaptured regions, but
  the visible costs still point at MoE experts, Q/K RMS scheduling, and
  prefill-shaped TP allreduce as the next code-level optimization targets.

See `notes/2026-05-14-minimax-compiled-path-repair.md` for the active repair
matrix and exact JSON/log paths.
