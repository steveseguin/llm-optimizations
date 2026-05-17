# MiniMax Current Path Repeatability

Date: 2026-05-17

## Result

I reran the current promoted MiniMax TP4 path three times after a single
adjacent control reached `61.894` output tok/s. The repeat sweep shows that
small one-run gains are not enough to promote a result.

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- p512 / n1536 / batch 1 / context 2048 / block size 256

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| 1 | 61.402229 | 81.869639 |
| 2 | 60.660075 | 80.880100 |
| 3 | 61.552868 | 82.070491 |
| mean | 61.205057 | 81.606743 |

The previously promoted strict baseline remains the cleaner public number:

- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`

## Interpretation

For this setup, single-run deltas below about `1` output tok/s are not reliable
enough to claim a real improvement. Future candidates should use adjacent
control/candidate pairs and repeats before being promoted or submitted to
LocalMaxxing.

## Artifacts

- `data/minimax-m27-current-promoted-repeatability-20260517.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/current-promoted-repeat-20260517/summary.json`
