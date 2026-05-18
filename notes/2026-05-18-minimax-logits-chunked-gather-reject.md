# MiniMax Logits Chunked Gather Rejection

Date: 2026-05-18

## Candidate

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Baseline recipe: FP16 activations, AutoRound INT4 W4A16, default XPU FlashAttention v2, XPU PIECEWISE graph, exact MiniMax router-logits path feeding llm-scaler INT4 MoE work-sharing decode
- Candidate flag: `VLLM_XPU_LOGITS_CHUNKED_GATHER=32768`
- Intent: split final TP logits all-gather into smaller chunks to see whether final logits communication can be reduced or scheduled more favorably without changing token selection.

## Quality Result

The candidate passed the first three strict gates:

- raw145 n64 exact token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: passed

It failed the 16-repeat arithmetic gate:

- Gate status: failed
- Failure reason: `nondeterministic lstrip_text`
- Combined arithmetic token hash: `7145a9fe4a4d482ca642e391d9c18c3f0c0b222bdcd027e1d488c30cd3952ada`
- Run hashes: 15 repeats produced `def6899500b2364bc97d561fc5f9cc78aa9fbcd5a0eb032eab1f2c6735d2bbec`; one repeat produced `9409e53d9c5444f8e179bee4951544a7b36986e5d53a0d90aca0a0479ecdecad`
- NUL/control output: none

## Decision

Reject without benchmarking. The output stayed semantically close enough to include `42`, but the strict repeatability gate caught a real token-level nondeterminism. This is not quality-safe, and no throughput result from this candidate should be promoted or submitted to LocalMaxxing.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-chunked-gather-32768-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T143607Z-summary.json`
- Quality artifacts:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-chunked-gather-32768-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T143607Z-quality/raw145-n64-exact.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-chunked-gather-32768-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T143607Z-quality/raw145-n256-exact.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-chunked-gather-32768-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T143607Z-quality/semantic-suite-n64-r2.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-chunked-gather-32768-logits-ws-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T143607Z-quality/arithmetic-repeat-n64-r16.json`
