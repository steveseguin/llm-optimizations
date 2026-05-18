# MiniMax Logits-WS MoE-Delay Retest Negative, 2026-05-18

## Goal

Retest `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1` on top of the current promoted
MiniMax logits-to-work-sharing path. The older MoE-delay result predated the
strict logits-WS promotion, so this run checks whether delaying the post-MoE
tensor-parallel allreduce becomes useful once router logits feed the llm-scaler
work-sharing INT4 MoE kernel directly.

## Recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Shape: p512/n1536, ctx2048, batch 1
- Backend: default XPU FlashAttention v2, XPU PIECEWISE graph
- Baseline flags:
  - `VLLM_XPU_USE_LLM_SCALER_MOE=1`
  - `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
  - `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
  - `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
  - `VLLM_XPU_ENABLE_XPU_GRAPH=1`
  - `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
  - `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
  - `CCL_TOPO_P2P_ACCESS=1`
- Candidate flag:
  - `VLLM_MINIMAX_MOE_DELAY_ALLREDUCE=1`

## Quality

The candidate passed the full strict quality gate before benchmarking:

- raw145 n64 exact token hash matched:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash matched:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite passed
- arithmetic repeat passed 16/16 deterministic exact `42`
- extended sixpack passed

## Performance

Current promoted logits-WS baseline:

- `81.758267` output tok/s
- `109.011023` total tok/s

MoE-delay retest on top of logits-WS:

| run | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | `80.747011` | `107.662681` |
| 2 | `77.291991` | `103.055988` |
| mean | `79.019501` | `105.359335` |

Delta versus promoted baseline: about `-3.35%` output throughput.

## Decision

Do not promote and do not submit to LocalMaxxing. The delayed MoE allreduce path
is quality-safe in this harness, but it is slower than the promoted exact
logits-to-work-sharing baseline.

The useful learning is that simply moving the post-MoE allreduce into the next
residual add is not enough. The next useful MoE path should either reduce the
actual collective cost inside the work-sharing kernel/epilogue or change the
collective shape. Coarse delayed residual handling has now lost both before and
after the logits-WS promotion.

## Artifacts

- Summary JSON:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-moe-delay-strict-retest-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T161707Z-summary.json`
- Published data copy:
  `data/minimax-m27-logits-ws-moe-delay-negative-20260518.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-logits-ws-moe-delay-strict-retest-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T161707Z-quality`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T163255Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T163543Z.json`
- Bench logs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T163255Z.log`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T163543Z.log`
