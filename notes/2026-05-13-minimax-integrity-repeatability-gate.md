# MiniMax Integrity and Repeatability Gate, 2026-05-13

## Purpose

Add a promotion gate for MiniMax M2.7 AutoRound results so future performance
claims are repeatable and quality-preserving enough to share. This gate is for
the current 4x B70 TP4 recipe:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- engine: vLLM `0.20.1-local`, XPU/Level Zero
- recipe: TP4, FP16, llm-scaler INT4 MoE, XPU graph, graph partition,
  MiniMax attention delayed allreduce, `--block-size 256`,
  `MAX_BATCHED_TOKENS=512`, `--no-enable-prefix-caching`
- AOT hash: `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`

## New Tools

- `scripts/validate-minimax-aot-collectives.py`
  - Validates the compiled AOT graph still has the expected 1496 allreduces,
    1496 waits, and MiniMax categories:
    - 8 embedding hidden allreduces
    - 496 Q/K RMS variance allreduces
    - 496 attention output projection hidden allreduces
    - 496 MoE hidden allreduces
- `scripts/run-vllm-minimax-quality-check.py`
  - Now mirrors the current recipe and supports repeated fixed greedy prompts.
  - Records token/text hashes and fails on nondeterministic token hashes.
- `scripts/summarize-vllm-repeatability.py`
  - Computes mean, median, min, max, stddev, and coefficient of variation for
    repeated vLLM JSON benchmark files.
- `scripts/run-minimax-current-best-integrity-gate.sh`
  - Wrapper for AOT collective validation, quality smoke, repeatability, and
    optional prefill screening.

## Gate Results

Quality smoke:

- graph path: pass
- prompts: 3 fixed prompts
- runs per prompt: 2
- deterministic token hashes: yes
- combined token hash:
  `08fadfadc952da331e999d14684ac797d0dd4382eb9644722c2b08573604dccf`
- combined text hash:
  `ac7df251c86427aefbf8843973a3e1ed553ff41f140ba2d86f504af91e7a0d48`

AOT collective integrity:

- pass
- actual allreduce calls: 1496
- actual wait calls: 1496
- wait gap: `{ "2": 1496 }`
- Q/K RMS variance allreduces: 496

Repeatability, p512/n1536, 3 independent measured runs:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| 1 | 73.127919 | 97.503892 |
| 2 | 73.369261 | 97.825682 |
| 3 | 73.235285 | 97.647047 |

Summary:

- output tok/s mean: `73.244155`
- output tok/s median: `73.235285`
- output tok/s min/max: `73.127919` / `73.369261`
- output tok/s stddev: `0.120915`
- output tok/s CV: `0.165085%`
- total tok/s mean: `97.658874`

The repeatability gate passed the thresholds:

- minimum run must be at least 98.5% of the promoted `73.306312` output tok/s
- output tok/s CV must be no more than 1.0%

## Interpretation

The existing LocalMaxxing submission at `73.306312` output tok/s is supported
as honest and repeatable. The new single-run high of `73.369261` is only
`0.086%` above the submitted value, while the 3-run mean is `73.244155`, so it
should be treated as normal variance rather than a new optimization. No new
LocalMaxxing submission was made.

The quality gate is deliberately conservative. It proves deterministic greedy
outputs for fixed prompts and verifies that the Q/K RMS allreduces remain in the
compiled graph. It does not replace full evals such as MMLU, coding, or agentic
quality checks.

## Artifacts

- Data summary:
  `data/minimax-m27-integrity-repeatability-gate-20260513.json`
- Quality gate:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/current-best-gate-20260513T214224Z`
- Repeatability gate:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/current-best-gate-20260513T214452Z`
- Benchmark JSONs:
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T214452Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T214729Z.json`
  - `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T215007Z.json`

## Next

Use this gate before promoting future MiniMax optimizations. Prefill should be
screened separately with longer prompts, but any prefill-oriented setting change
must rerun the p512/n1536 decode gate before it is treated as a new default.
