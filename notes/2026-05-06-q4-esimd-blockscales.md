# 2026-05-06 Q4_0 ESIMD Block-Loaded Scales

## Goal

Continue the standalone Q4_0 x Q8_1 ESIMD GEMV prototype and find a concrete kernel-level change that improves B70 speed before attempting llama.cpp integration.

## Change

Added two compile-time harness knobs:

- `Q4_ESIMD_BLOCK_LOAD_SCALES=1`: block-load the four Q4_0 per-block scales and eight Q8_1 scale/sum values once per 128-column ESIMD tile.
- `Q4_ESIMD_BIAS_IN_ACC=1`: fold the Q8 sum bias term into accumulator lanes instead of maintaining a scalar bias chain.

The block-scale change was suggested by read-only source review: the payload path already used ESIMD block loads, but scale metadata was still scalar-loaded in the unrolled inner loop.

## Results

All runs used one B70 via `ONEAPI_DEVICE_SELECTOR=level_zero:2`, 260 timed iterations, 40 warmups, and deterministic random data. Lower `median_us` is better.

| Shape | Mode | Baseline us | Block-scales us | Delta |
| --- | --- | ---: | ---: | ---: |
| `N=17408 K=5120` | single | 127.916 | 100.104 | +21.74% |
| `N=17408 K=5120` | fused2 | 191.145 | 177.708 | +7.03% |
| `N=4352 K=5120` | single | 46.042 | 28.021 | +39.14% |
| `N=4352 K=5120` | fused2 | 55.625 | 44.375 | +20.23% |
| `N=5120 K=17408` | single | 140.313 | 109.375 | +22.05% |
| `N=5120 K=1536` | single | 16.355 | 10.730 | +34.39% |
| `N=1024 K=5120` | single | 11.667 | 10.104 | +13.40% |
| `N=256 K=5120` | single | 11.980 | 10.209 | +14.78% |

Correctness stayed within the existing harness tolerance. Printed `max_abs` rounded to `0.000` in these runs.

Artifacts:

- main matrix TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/esimd-blockscales-20260506/esimd-blockscales-matrix-20260506T230220Z.tsv`;
- bias interaction TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/esimd-blockscales-20260506/esimd-blockscales-biasacc-matrix-20260506T230258Z.tsv`;
- BMG target screen TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/esimd-bmg-target-20260506/esimd-bmg-target-20260506T230103Z.tsv`.

## Side Screens

`Q4_ESIMD_BIAS_IN_ACC=1` by itself did not meet the acceptance bar: neutral on large single, slower on fused2/down/output shapes, and only one shard-sized single shape showed a small one-off win.

With block-loaded scales already enabled, bias-in-acc was still generally neutral or slower, except for one noisy `4352x5120` single run. Keep the macro off by default.

The direct `-fsycl-targets=intel_gpu_bmg_g31` build was not a general win versus `spir64_gen -Xs "-device bmg"`: much slower on large `17408x5120` single, roughly neutral on most shapes, and faster only on small `5120x1536`.

## Interpretation

Block-loading scale metadata is the first ESIMD harness change from this pass that looks strong enough to carry forward. It directly targets scalar metadata load/address overhead and improves every tested shape.

Next step is still not immediate llama.cpp integration. The harness kernel is standalone and uses a simplified layout. The practical next move is to port this scale-load pattern into a llama.cpp-compatible experimental ESIMD MMVQ path or add a microbenchmark that compares it against the exact current reordered MMVQ data layout.

