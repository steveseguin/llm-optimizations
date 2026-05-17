# MiniMax Deferred Output Copy Candidate

Date: 2026-05-17

## Result

The built-in XPU deferred output-copy path is quality-safe on the first exact
canary, but it is not faster than the promoted strict MiniMax baseline.

Runtime flags:

```bash
export VLLM_XPU_DEFER_ASYNC_OUTPUT_COPY=1
export VLLM_XPU_DEFERRED_COPY_SYNC_DEVICE=0
```

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- p512 / n1536 / batch 1 / context 2048

Quality screen:

- raw145 n64 exact token hash passed
- expected hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| repeat 1 | 61.182195 | 81.576260 |
| repeat 2 | 61.113440 | 81.484586 |
| mean | 61.147817 | 81.530423 |

Promoted strict baseline:

- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

Decision: do not promote and do not submit to LocalMaxxing. The candidate is a
valid learning, but it does not improve the current best.

## Artifacts

Result data:

- `data/minimax-m27-deferred-output-copy-no-improvement-20260517.json`

Quality:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/deferred-output-copy-screen/raw145-n64-defer-copy-sync0-20260517T183414Z.json`

Benchmarks:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/deferred-output-copy-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T184013Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/deferred-output-copy-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T184306Z.json`

## Lesson

The sampled-token CPU handoff is real, but simply deferring the copy and
skipping the explicit XPU synchronize did not help. The async baseline already
overlaps the tiny token copy reasonably well. The next candidate is a
default-off reusable pinned output-copy buffer to avoid per-step CPU tensor
allocation in `AsyncGPUModelRunnerOutput`.
