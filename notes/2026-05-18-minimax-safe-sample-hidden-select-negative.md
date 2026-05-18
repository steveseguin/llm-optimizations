# MiniMax Safe Sample Hidden Select Negative

Date: 2026-05-18

## Candidate

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode
- Candidate flag: `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT=1`
- Intent: avoid PyTorch advanced indexing for sampled hidden-state selection on XPU when the requested logits rows are already the full hidden-state batch; otherwise use `torch.index_select`.

## Quality Gate

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: passed and deterministic across two greedy repeats
- arithmetic repeat: passed 16 repeated greedy calls
- extended sixpack: passed and deterministic across two greedy repeats

## Result

- Repeat 1: `77.377158` output tok/s, `103.169544` total tok/s
- Repeat 2: `77.251550` output tok/s, `103.002066` total tok/s
- Mean: `77.314354` output tok/s, `103.085805` total tok/s
- Current promoted baseline: `81.758267` output tok/s, `109.011023` total tok/s

## Decision

Reject for promotion. The flag is quality-clean but roughly 5.4% slower than the promoted logits-WS baseline on the p512/n1536 decode benchmark. It should not be submitted to LocalMaxxing as an achievement result.

The strict runner was updated after this run to include `VLLM_XPU_SAFE_SAMPLE_HIDDEN_SELECT` in future `candidate_env` summaries. The generated summary for this specific run omitted the flag, but the invocation had it set.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-safe-sample-hidden-select-logits-ws-strict-tp4-ctx2048-mbt512-bs256-20260518T133805Z-summary.json`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T135400Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T135651Z.json`
