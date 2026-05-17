# MiniMax Local-Argmax Tight Quality Refresh

Date: 2026-05-17

## Summary

The MiniMax M2.7 AutoRound local-argmax path has a stricter quality-promoted
refresh. This run used a fresh AOT cache root, required the installed vLLM
runtime path and local-argmax pair-all-gather marker, tightened the arithmetic
canary to reject quoted or approximate answers, and ran extended gates before
benchmarking.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Shape: prompt 512, output 1536, context 2048, batch 1
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

## Result

Two throughput repeats after quality gates:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 61.250688 | 81.667584 |
| 2 | 61.557382 | 82.076509 |
| mean | 61.404035 | 81.872046 |

The vLLM benchmark JSON reports total tokens per second over 512 prompt tokens
plus 1536 output tokens. It does not separately expose prefill tok/s for this
run, so `tokSTotal` is the reproducible prompt+decode metric recorded here.

## Quality Gates

All gates passed before benchmarking:

- raw145 n64 exact token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS, exact arithmetic `42`, `add_one` code canary
- arithmetic repeat: exact `42`, 16 greedy repeats
- extended sixpack: PASS, arithmetic, code, JSON, sort, SQL

The arithmetic prompt was tightened after a graph/clone isolation run returned
both `42` and `"~42"` under the old prompt. The new canary requires exactly
`42` via prompt wording and prompt-scoped regex, so formatting drift is no
longer accepted as a semantic pass.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-tightquality-extended-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T151011Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-tightquality-extended-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T151011Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T152524Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T152816Z.json`
- LocalMaxxing payload:
  `data/localmaxxing-minimax-m27-autoround-localargmax-tightquality-p512n1536-20260517.payload.json`
- LocalMaxxing response:
  `data/localmaxxing-responses/minimax-m27-autoround-localargmax-tightquality-p512n1536-20260517.response.json`

## Decision

Promote this as the current strict MiniMax baseline. It is only a small speed
increase over the prior runtime-guarded `61.317497` tok/s row, but it is more
defensible because it uses the tightened canary, repeat16 arithmetic gate, and
extended sixpack before benchmarking.

The next performance work remains the same: reduce the pair all-gather and
framework sampling overhead without changing greedy target semantics. Candidate
paths still need the full strict quality gate before any LocalMaxxing update.
