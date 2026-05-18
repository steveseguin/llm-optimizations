# MiniMax Safe Sample Hidden Select Neutral

Date: 2026-05-18

## Candidate

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode
- Candidate flag: `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1`
- Intent: avoid PyTorch advanced indexing for sampled hidden-state selection on XPU when the requested logits rows are already the full hidden-state batch; otherwise use `torch.index_select`.

## Erratum

The first recorded safe-selector run used the strict runner's older `TRITON_ATTN` default, while the promoted baseline used default XPU FlashAttention v2. That first run passed quality, but its `77.314354` output tok/s result is not a fair comparison to the promoted `81.758267` output tok/s FlashAttention baseline.

The strict runner has since been changed to default to the same FlashAttention backend used by the promoted baseline. The fair rerun below is the result to compare.

## Quality Gate

The fair FlashAttention rerun passed the full strict quality gate before benchmarking:

- raw145 n64 exact token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: passed and deterministic across two greedy repeats
- arithmetic repeat: passed 16 repeated greedy calls
- extended sixpack: passed and deterministic across two greedy repeats

The benchmark logs confirm default XPU FlashAttention v2:

- `Using Flash Attention backend.`
- `Using FlashAttention version 2`

## Fair FlashAttention Result

- Repeat 1: `81.961290` output tok/s, `109.281721` total tok/s
- Repeat 2: `81.867044` output tok/s, `109.156059` total tok/s
- Mean: `81.914167` output tok/s, `109.218890` total tok/s
- Current promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s
- Delta: `+0.19%` output tok/s, within normal run variance

## Backend-Mismatched Diagnostic Result

- Backend: `TRITON_ATTN`, not the promoted FlashAttention backend
- Repeat 1: `77.377158` output tok/s, `103.169544` total tok/s
- Repeat 2: `77.251550` output tok/s, `103.002066` total tok/s
- Mean: `77.314354` output tok/s, `103.085805` total tok/s
- Use: diagnostic only; do not compare this directly against the promoted FlashAttention baseline

## Decision

Do not promote and do not submit to LocalMaxxing. The candidate is quality-clean under the fair backend, but the `+0.19%` mean output delta is too small to distinguish from noise. It is best recorded as a neutral/tie result, not an achievement.

## Artifacts

Fair FlashAttention rerun:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-safe-sample-hidden-select-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T140741Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T142318Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T142609Z.json`
- Bench logs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T142318Z.log`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T142609Z.log`

Backend-mismatched diagnostic run:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-safe-sample-hidden-select-logits-ws-strict-tp4-ctx2048-mbt512-bs256-20260518T133805Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T135400Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T135651Z.json`
