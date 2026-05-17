# MiniMax Local-Argmax XPU Reduce: Quality Pass, No Uplift

Date: 2026-05-17

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM `0.20.1-local`, XPU, TP4

## Purpose

Test whether the strict greedy local-argmax tail can be sped up by keeping the
existing TP pair all-gather but replacing the PyTorch pair reduction with a tiny
SYCL reducer:

```bash
VLLM_XPU_LOCAL_ARGMAX_XPU_REDUCE=1
```

This is intentionally smaller than a full logits or MoE fusion. The goal was to
remove part of the CPU/framework tail without changing logits, sampling,
quantization, routing, Q/K variance allreduce, or model quality.

## Standalone Probe

Artifacts:

- Collective probe: `/home/steve/bench-results/minimax-m2.7-strict-candidates/pair-collective-probe-single-20260517T203739Z`
- XPU reducer probe: `/home/steve/bench-results/minimax-m2.7-strict-candidates/pair-argmax-xpu-reduceonly-oraclefix-20260517T205352Z`

Findings:

- `all_gather_into_tensor` was correct in the standalone pair probe and measured
  about `0.118-0.125 ms` for batch 1.
- `all_gather_list` was correct but slower at about `0.238-0.240 ms`.
- `gather_broadcast` and `all_to_all_repeated` timed out or hung.
- The full helper path that used c10d functional `all_gather_into_tensor` inside
  the extension failed the corrected `rank_wins` oracle, so it was not wired
  into vLLM.
- The reduce-only helper, `reduce_flat_pairs`, passed `rank_wins`, `mixed_sign`,
  `negative_only`, `token_tie`, and `random`.

## Strict Gate

The vLLM candidate passed the strict quality gate before benchmarking:

| Check | Result | Token hash |
| --- | --- | --- |
| raw145 n64 exact | pass | `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd` |
| raw145 n256 exact | pass | `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537` |
| semantic suite n64/r2 | pass | `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805` |
| arithmetic repeat n64/r8 | pass | `261779104d5abf1642713bfc560ca8d2d6c0f16edbcc929c8b0819b5a760dd7c` |

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-xpu-reduce-screen-strict-tp4-ctx2048-mbt512-bs256-20260517T205612Z-summary.json
```

## Benchmark

Shape:

- p512/n1536
- batch 1
- context 2048
- TP4
- block size 256
- `MAX_BATCHED_TOKENS=512`

Result:

| Path | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| promoted strict baseline | `61.404035` | `81.872046` |
| XPU reduce candidate | `60.071619` | `80.095493` |

Benchmark artifacts:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T210910Z.json`
- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T210910Z.log`

## Decision

Rejected as a speed path. The candidate is quality-safe, but it is below the
current promoted strict result and within the run-to-run variance measured
earlier today. It was not submitted to LocalMaxxing.

Keep `VLLM_XPU_LOCAL_ARGMAX_XPU_REDUCE` unset for real MiniMax runs.

The result is still useful pruning data: the pair-reduction tail is too small to
be the main bottleneck. The next optimization work should stay focused on MoE
expert dispatch, Q/K RMS variance allreduce plus RMS application, KV attention,
RoPE scheduling, and residual allreduce fusion.

## Published Source

- `experiments/minimax_pair_argmax_xpu/`
- `benchmarks/b70_pair_argmax_xpu_probe.py`
- `data/minimax-m27-xpu-reduce-localargmax-no-uplift-20260517.json`
