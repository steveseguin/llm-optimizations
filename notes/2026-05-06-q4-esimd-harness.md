# 2026-05-06 - Q4_0 x Q8_1 ESIMD harness

## Context

Goal: port the useful idea from llm-scaler INT4 GEMV into a llama-compatible Q4_0 x Q8_1 decode harness before touching llama.cpp graph execution.

Source:

`tools/q4_0_gemv_esimd_bench.cpp`

Build:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -std=c++17 -O3 -fsycl -ffast-math \
  -fsycl-device-code-split=per_kernel \
  -fsycl-targets=spir64_gen -Xs "-device bmg" \
  tools/q4_0_gemv_esimd_bench.cpp \
  -o tools/q4_0_gemv_esimd_bench
```

## What It Tests

- Reordered Q4_0 SoA layout:
  - `q4_qs`: `[N, K/2] uint8`
  - `q4_d`: `[N, K/32] half`
- Reordered Q8_1 activation:
  - `q8_qs`: `[K] int8`
  - `q8_d`: `[K/32] half`
  - `q8_s`: `[K/32] half`
- Output: `[N] float`
- Formula matches GGML Q4_0 x Q8_1:

`d4 * (d8 * dot(q4_unsigned, q8_qs) - 8 * q8_sum)`

The first prototype supports ESIMD `K_SPLIT` values `1`, `2`, `4`, and `8` with SLM reduction.

## Results

All runs used one B70 via `ONEAPI_DEVICE_SELECTOR=level_zero:0`.

| Shape | Meaning | Best ks | Best us | Approx GB/s | Error |
| --- | --- | ---: | ---: | ---: | --- |
| `N=17408 K=5120` | FFN gate/up full tensor | 2 | 125.134858 | 401.250560 | max abs `0.000023` |
| `N=5120 K=17408` | FFN down full tensor | 2 | 121.849692 | 411.778670 | max abs `0.000057` |
| `N=1024 K=5120` | attn k/v full tensor | 2 | 10.132800 | 292.019580 | max abs `0.000015` |
| `N=256 K=5120` | attn k/v 4-way shard | 4 | 4.363105 | 170.535433 | max abs `0.000011` |
| `N=4352 K=5120` | FFN gate/up 4-way shard | 4 | 31.106260 | 403.678488 | max abs `0.000023` |
| `N=5120 K=4352` | FFN down 4-way shard | 1 | 29.260755 | 429.214352 | max abs `0.000008` |
| `N=5120 K=1536` | attn output shard | 1 | 14.215855 | 312.741513 | max abs `0.000004` |

## Interpretation

- Correctness is good enough for continued kernel work.
- The first ESIMD prototype is not yet a clear integration win. Effective bandwidth is only around `300-430 GB/s` for the large shapes.
- K-split helps some shapes:
  - Large full FFN shapes prefer `ks=2`.
  - `N=4352 K=5120` shard prefers `ks=4`.
  - `N=5120 K=4352` and `N=5120 K=1536` prefer `ks=1`.
- The next kernel step should optimize memory access and accumulation before integrating into llama.cpp. Candidate changes:
  - use packed integer dot products rather than float unpack/FMA for Q4/Q8 inner products,
  - process multiple output rows per ESIMD work-item where N is large,
  - specialize shapes instead of a single generic kernel,
  - compare against current llama MMVQ with a controlled microbenchmark before graph integration.

## Decision

Do not wire this prototype into llama.cpp yet. Keep it as a reproducible correctness and timing harness, then iterate on the kernel until it plausibly beats the current Q4_0 MMVQ path.
