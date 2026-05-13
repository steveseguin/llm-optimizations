# MiniMax Fused GEMV And Runtime Screens, 2026-05-13

Goal: continue pushing MiniMax M2.7 AutoRound W4A16 toward `60+` output tok/s
without changing model quality. The active quality path remains TP4, FP16,
llm-scaler u4 MoE decode, static single-token AOT graph, and `--async-engine`.

## End-To-End Screens

| Test | Prompt/Output | Output tok/s | Total tok/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| async static graph + `--gpu-memory-utilization 0.95` | 512/1536 | 48.42 | 64.55 | not repeated |
| same repeat | 512/1536 | 46.21 | 61.61 | variance/negative |
| `mode=3` + graph partition + `compile_sizes=[1]` | 512/512 | 33.24 | 66.48 | negative cold-cache artifact |

`gpu_memory_utilization=0.95` raises KV capacity as expected (`32,256` then
`33,024` GPU KV-cache tokens) but did not repeat above the accepted
`48.092807` output tok/s result. Do not submit the first run as a new record.

The `mode=3` compile screen produced the same AOT hash as the current best
(`3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`) after
about `178.90 s` of compile work, then hit the known cold-cache symptom:
`9,408` GPU KV-cache tokens and only `33.24` output tok/s at p512/n512. This is
not a speed path.

## Fused ResAdd/RMS/INT4 GEMV Probe

I tested llm-scaler core `esimd_resadd_norm_gemv_int4_pert` as a possible
MiniMax dense-projection fusion. The actual TP4 shapes are:

- `qkv_proj`: `N=2048,K=3072`
- `o_proj`: `N=3072,K=1536`

The probe found a correctness hazard in the fused kernel. It launches one
workgroup per output row, but only output-row workgroup 0 writes the updated
`residual` and `normed_out` while peer workgroups may still be reading
`residual`. That is a cross-workgroup race.

Evidence:

- Current fused kernel at `N=3072,K=1536`: `fused_rel=0.10329`,
  `fused_vs_vllm_max=36.10`.
- A temporary local no-store diagnostic build removed those residual/normed
  writes and reduced the same-shape fused error to `rel=0.00048`, confirming
  the race.
- Once the unsafe writes were removed, the actual `o_proj` shape was slower
  than oneDNN INT4-only (`64.75 us` fused no-store versus `53.30 us` oneDNN
  INT4-only in the synthetic probe), so this does not justify a vLLM integration.
- The source and core shared object were restored to original semantics after
  the diagnostic build.

Repro script:

```bash
source /home/steve/.venvs/vllm-xpu/bin/activate
PYTHONPATH=/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python:$PYTHONPATH \
  python /home/steve/llm-optimizations-publish/benchmarks/b70_resadd_norm_gemv_int4_race_probe.py --json
```

## Decision

Do not pursue `esimd_resadd_norm_gemv_int4_pert` as a drop-in MiniMax
projection fusion. The race is fixable by separating residual/norm production
from GEMV or redesigning the launch topology, but then the kernel loses the
apparent speed advantage on the real `o_proj` shape.

The 60 tok/s path remains real collective/epilogue fusion around existing graph
wait sites, not this standalone fused GEMV helper.

## Logs

- gmem first run:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-async-staticcompile-gmem095-p512n1536-20260513T011847Z`
- gmem repeat:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-async-staticcompile-gmem095-repeat-p512n1536-20260513T012136Z`
- mode3 compile screen:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-mode3-partition-compile1-p512n512-20260513T012448Z`
- no-store diagnostic build:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/resadd-norm-gemv-coreonly-build-20260513T011248Z.log`
- restored core build:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/resadd-norm-gemv-coreonly-force-restore-build-20260513T011644Z.log`
