# MiniMax Safe Hidden Selection Repeatability

Date: 2026-05-18

## Candidate

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode
- Candidate flag: `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1`
- Shape: p512/n1536, ctx2048, batch 1, MBT512, block256

This repeated the earlier safe hidden-state selection screen with four benchmark repeats after the full strict quality gate. The goal was to determine whether the prior `+0.19%` result was real or just normal run variance.

## Quality Gate

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: passed and deterministic across two greedy repeats
- arithmetic repeat: passed 16 repeated greedy calls
- extended sixpack: passed and deterministic across two greedy repeats

## Repeatability Result

- Repeat 1: `81.465671` output tok/s, `108.620894` total tok/s
- Repeat 2: `80.306714` output tok/s, `107.075619` total tok/s
- Repeat 3: `81.434420` output tok/s, `108.579226` total tok/s
- Repeat 4: `82.311165` output tok/s, `109.748220` total tok/s
- Four-repeat mean: `81.379492` output tok/s, `108.505990` total tok/s
- Current promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s
- Delta: `-0.46%` output tok/s versus promoted

## Decision

Do not promote and do not submit to LocalMaxxing. The candidate is quality-clean, but the four-repeat run lands below the current promoted MiniMax result. The earlier `+0.19%` screen should be treated as noise rather than a reliable improvement.

This reinforces that the hidden-state selection change is safe but not worth carrying as a promoted runtime flag. Next useful work should focus on decode-critical collective boundaries and fused epilogues rather than more Python-side selection reties.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-safe-hidden-repeat4-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T164551Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T170139Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T170430Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T170719Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T171016Z.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-safe-hidden-repeat4-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T164551Z-quality`
