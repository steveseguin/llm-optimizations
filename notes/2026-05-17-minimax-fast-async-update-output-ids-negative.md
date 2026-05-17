# MiniMax Fast Async Update-Output-IDs Candidate

Date: 2026-05-17

## Result

The async scheduler `update_async_output_token_ids()` batch-1 list bypass is
quality-safe on the exact canary, but it does not improve throughput.

Runtime flag:

```bash
export VLLM_XPU_FAST_ASYNC_UPDATE_OUTPUT_IDS=1
```

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- p512 / n1536 / batch 1 / context 2048 / block size 256

Quality screen:

- raw145 n64 exact token hash passed
- expected and observed hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| adjacent control | 61.433926 | 81.911902 |
| fast update output ids | 61.265723 | 81.687631 |
| delta | -0.168203 | -0.224271 |

Decision: do not promote and do not submit to LocalMaxxing. The same-day
current-path sweep showed `60.660` to `61.553` output tok/s over three repeats,
so this candidate is inside normal variance and slower than adjacent control.

## Artifacts

Result data:

- `data/minimax-m27-fast-async-update-output-ids-negative-20260517.json`

Patch:

- `patches/minimax-fast-async-update-output-ids-negative-20260517.patch`

Quality:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-update-output-ids-screen/raw145-n64-fast-async-update-output-ids-20260517T200933Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-update-output-ids-screen/raw145-n64-fast-async-update-output-ids-20260517T200933Z.log`

Benchmarks:

- control:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-update-output-ids-bench/control/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T201504Z.json`
- candidate:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/fast-async-update-output-ids-bench/candidate/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T201215Z.json`

## Lesson

The generic `.tolist()` call in `update_async_output_token_ids()` is not a
dominant decode bottleneck on this recipe. Continue focusing on lower-level
GPU work: graph coverage, MiniMax MoE/logits path, attention collectives, and
larger host/device framework boundaries.
