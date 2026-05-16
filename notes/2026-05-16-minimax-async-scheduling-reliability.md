# MiniMax M2.7 Async Scheduling Reliability Gate

Date: 2026-05-16

Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM 0.20.1-local XPU with llm-scaler INT4 MoE path

## Summary

The no-clone compile allreduce path remains the best performance-oriented XPU
graph candidate, but stricter multi-prompt quality testing found a reliability
problem in the broader async scheduling graph recipe. The issue is not caused
by `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`: the older clone/copy-back path
failed the same six-prompt pack in the same way.

The quality-safe recipe is to keep the no-clone compile allreduce path and
disable vLLM async scheduling:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
vllm bench throughput ... --no-async-scheduling ...
```

This reduces the p512/n1536 mean from the earlier 66.28 output tok/s result to
62.66 output tok/s, but it passes the stricter reliability gate and still clears
the 60 tok/s target.

## Evidence

Accepted no-clone result before this pass:

- Mean p512/n1536 output tok/s: 66.28
- Strict raw145 n64/n256 and semantic gate: pass
- Extended six-prompt pack had not yet been promoted to a hard gate

New reliability probes:

- PASS-only no-clone, 5 repeats: pass
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-reliability-no-clone-pass-only-r5-20260516T023607Z.json`
- PASS+SQL no-clone, 3 repeats: pass
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-reliability-no-clone-pass-sql-r3-20260516T024531Z.json`
- Six-prompt pack, no-clone, async scheduling enabled: fail
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-extended-raw-pack-no-clone-v2-20260516T023041Z.json`
  - Failure: second repeat, first PASS prompt generated 64 token-id-0 NUL tokens
- Six-prompt pack, older clone/copy-back path, async scheduling enabled: fail
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-reliability-clone-sixpack-v2-r2-20260516T025347Z.json`
  - Failure: same second-repeat PASS NUL pattern
- Six-prompt pack, no-clone, async scheduling disabled: pass
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-reliability-no-clone-sixpack-v2-asyncoff-r2-20260516T030123Z.json`
  - Combined token hash: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`
- Raw145 n256 exact hash, no-clone, async scheduling disabled: pass
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-reliability-no-clone-raw145-n256-asyncoff-20260516T031651Z.json`
  - Combined token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`

Throughput with no-clone plus `--no-async-scheduling`:

- Run 1: 62.6668 output tok/s, 83.5557 total tok/s
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260516T031032Z.json`
- Run 2: 62.6578 output tok/s, 83.5438 total tok/s
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260516T031317Z.json`
- Mean: 62.6623 output tok/s, 83.5497 total tok/s
- LocalMaxxing accepted result id: `cmp7sf8xo00dko4018nmrmckc`
  - Payload: `data/localmaxxing-minimax-m27-autoround-no-clone-asyncoff-p512n1536-20260516.payload.json`
  - Response: `data/localmaxxing-responses/minimax-m27-autoround-no-clone-asyncoff-p512n1536-20260516.response.json`

## Decision

Use async scheduling disabled for quality-sensitive single-session MiniMax M2.7
TP4 runs until the XPU async scheduling state transition is fixed or proven safe
by a stronger gate. The earlier 66.28 tok/s no-clone result remains a useful
performance datapoint, but it should be labeled as less reliability-cleared than
the 62.66 tok/s async-off recipe.

The next optimization path should keep async scheduling disabled while looking
for lower-level deterministic speedups:

- XPU Q/K RMS and allreduce fusion that preserves the existing fp32 variance
  allreduce semantics.
- Decode graph boundary reduction without `_functional_collectives`.
- MoE epilogue or routing-side improvements that do not alter expert selection
  or token outputs.
- A targeted vLLM/XPU async scheduling bug report or patch using the six-prompt
  pack as the repro.
