# MiniMax xpu-smi Live Trace, 2026-05-10

## Goal

Check whether MiniMax M2.7 AutoRound TP4 decode is obviously frequency- or
power-throttled on the four B70s.

`xpu-smi stats -j` without a device ID returns `{"error":"Unknow operation"}` on
this build, so the usable trace sampled each device separately with:

```bash
xpu-smi stats -d 0 -j
xpu-smi stats -d 1 -j
xpu-smi stats -d 2 -j
xpu-smi stats -d 3 -j
```

## Outcome

The per-device polling was too intrusive for a valid benchmark. During a
p512/n1536 MiniMax AutoRound reference run, vLLM emitted:

```text
No available shared memory broadcast block found in 60 seconds.
```

The run was killed and is not a throughput result.

The partial trace is still useful. From the decode/stall window starting around
`2026-05-10T13:09:41Z`, all four B70s reported:

| Device | Avg power W | Max power W | Frequency | Avg mem util |
| --- | ---: | ---: | --- | ---: |
| 0 | 94.90 | 96.33 | 2800 MHz | 96.03% |
| 1 | 114.04 | 115.71 | 2800 MHz | 95.32% |
| 2 | 115.02 | 116.47 | 2800 MHz | 95.31% |
| 3 | 96.43 | 97.89 | 2800 MHz | 95.32% |

Interpretation:

- no evidence of frequency throttling in the sampled window;
- memory is close to full in the default p512/n1536, max-model-len 2048 recipe;
- power is asymmetric, with middle ranks drawing around 115 W and outer ranks
  around 95-97 W, which fits a communication/scheduling imbalance more than a
  global power-limit problem;
- `xpu-smi stats` should not be polled during official benchmark runs on this
  stack because the management calls can perturb or stall vLLM TP4.

Next profiling should use a lower-overhead Level Zero metric path, a kernel-side
timer, or offline traces around specific custom kernels rather than live xpu-smi
polling.
