# MiniMax Logits-WS Q/K RMS Helper Negative, 2026-05-18

## Goal

Retest the MiniMax Q/K RMS XPU helper on top of the current strict promoted
logits-to-work-sharing MiniMax path. Older tests showed the helper was not a
speed path because it still leaves the tensor-parallel Q/K variance allreduce in
the middle of decode. This run checks whether that changes after the current
logits-WS MoE improvement and default XPU FlashAttention v2 recipe.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Shape: p512/n1536, ctx2048, batch 1
- Backend: default XPU FlashAttention v2, XPU PIECEWISE graph
- Baseline flags:
  - `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
  - `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
  - `VLLM_XPU_ENABLE_XPU_GRAPH=1`
  - `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
  - `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
  - `CCL_TOPO_P2P_ACCESS=1`
- Candidate flag:
  - `VLLM_MINIMAX_QK_RMS_XPU_HELPER=1`

## Quality

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact token hash matched:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash matched:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite passed
- arithmetic repeat passed 16/16 deterministic exact `42`
- extended sixpack passed

One shutdown cleanup emitted `Bad address (src/pipe.cpp:367)` after a completed
quality subprocess, but the strict summary remained `quality_passed` and later
gates continued cleanly. Treat this as a runtime cleanup warning, not a quality
failure.

## Performance

Current promoted logits-WS baseline:

- `81.758267` output tok/s
- `109.011023` total tok/s

Q/K RMS helper retest:

| run | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | `80.289833` | `107.053111` |
| 2 | `82.594022` | `110.125362` |
| mean | `81.441928` | `108.589237` |

Delta versus promoted baseline: about `-0.39%` output throughput.

## Decision

Do not promote and do not submit to LocalMaxxing. The helper is quality-safe in
this harness, but does not beat the current promoted logits-WS baseline.

The useful learning is unchanged: replacing only the local Q/K RMS math is not
enough. The decode-critical boundary is the Q/K variance collective plus the
post-collective RMS/apply placement. The next useful Q/K path needs a graph-safe
collective+RMS fusion or a deeper attention/KV scheduling change, not a separate
standalone helper wrapped around the same allreduce.

## Artifacts

- Summary JSON:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-qk-rms-helper-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T153811Z-summary.json`
- Published data copy:
  `data/minimax-m27-logits-ws-qk-rms-helper-negative-20260518.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-qk-rms-helper-flashdefault-strict-tp4-ctx2048-mbt512-bs256-20260518T153811Z-quality`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T155357Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T155647Z.json`
- Bench logs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T155357Z.log`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T155647Z.log`
