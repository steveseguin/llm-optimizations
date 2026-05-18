# MiniMax Cached MoE Op Lookup Screen

Date: 2026-05-18

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM `0.20.1-local`, XPU TP4

Baseline for comparison: current strict promoted logits-to-work-sharing no-attention-delay recipe, `82.404268` output tok/s and `109.872357` total tok/s mean at p512/n1536, ctx2048, MBT512, block256.

## Summary

This candidate cached the Python callable for the llm-scaler MiniMax MoE custom op after first import. It was intended to remove per-call dynamic import/callable lookup around the exact router-logits-to-work-sharing decode path.

The change was math-preserving and did not alter tensors, kernels, graph settings, or routing. It passed the full strict quality suite, but benchmarked slightly below the promoted baseline.

## Quality

All checks passed:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Benchmark

Shape: p512/n1536, ctx2048, batch 1, TP4, MBT512, block256.

Repeats:

- `82.070603` output tok/s, `109.427470` total tok/s
- `81.942496` output tok/s, `109.256661` total tok/s

Mean:

- `82.006549` output tok/s
- `109.342066` total tok/s

Delta vs current promoted baseline:

- Output tok/s: about `-0.48%`
- Total tok/s: about `-0.48%`

## Decision

Do not promote and do not submit to LocalMaxxing. The patch is quality-safe, but the result is below the promoted mean and within normal run variance.

The active vLLM source was restored to the promoted dynamic-import behavior after the screen so future candidates compare against the same baseline.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-cached-moe-op-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T201819Z-summary.json`
- Benchmark 1: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T203412Z.json`
- Benchmark 2: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T203702Z.json`
