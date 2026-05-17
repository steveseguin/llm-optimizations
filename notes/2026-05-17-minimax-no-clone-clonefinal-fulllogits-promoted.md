# MiniMax Full-Logits No-Clone + Final-Hidden Clone Promoted Result

Date: 2026-05-17

## Summary

This candidate restores the faster full-logits decode path while keeping the
`xpu_communicator` no-clone allreduce optimization, but adds a targeted
`VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1` guard. The intent was to test whether
the earlier full-logits no-clone quality failure was caused by final-hidden
aliasing rather than by the no-clone allreduce path itself.

Result: promoted. It passed the strict quality harness and improved decode
throughput over both the same-day local-argmax anchor and the previously
published strict LocalMaxxing baseline.

LocalMaxxing accepted the result as `cmpaepjz10043o101822jzc73`.

## Runtime

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- CPU/RAM: AMD EPYC 9015 8-Core Processor, 16 GiB RAM plus 64 GiB swap
- OS: Ubuntu 24.04.4 LTS
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16 / INC
- Shape: p512, n1536, ctx2048, batch 1
- Block size: 256
- Prefix cache: disabled
- Temperature: greedy / 0
- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1`
- `CCL_TOPO_P2P_ACCESS=1`
- No local-argmax path for this promoted run:
  `VLLM_XPU_LOCAL_ARGMAX_DECODE` unset

Representative launch:

```bash
LABEL=no-clone-clonefinal-fulllogits-bench \
BENCH_REPEATS=2 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-strict-no-clone-clonefinal-fulllogits-20260517 \
VLLM_RUNTIME_REQUIRE_ANY_MARKERS=VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE,VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN \
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1 \
VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1 \
VLLM_BENCH_TEMPERATURE=0 \
CCL_TOPO_P2P_ACCESS=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZE_AFFINITY_MASK=0,1,2,3 \
/home/steve/llm-optimizations-publish/scripts/run-minimax-strict-quality-gated-candidate.sh
```

## Quality Result

The run passed the full strict gate before benchmarking:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This is the key difference versus the older full-logits no-clone result. The
old row was fast but later failed extended quality. This run did not reproduce
that corruption under the current strict gate.

## Benchmark Result

Two p512/n1536 throughput repeats after quality gates:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 66.470145 | 88.626859 |
| 2 | 66.500566 | 88.667421 |
| mean | 66.485355 | 88.647140 |

Comparison:

- Same-day quality-safe local-argmax anchor:
  `60.642082` output tok/s, `80.856109` total tok/s
- Improvement versus same-day anchor:
  `+9.64%` output tok/s
- Previous promoted strict LocalMaxxing baseline:
  `61.404035` output tok/s, `81.872046` total tok/s
- Improvement versus previous promoted strict row:
  `+8.28%` output tok/s

## Decision

Promote this as the current best quality-safe MiniMax M2.7 AutoRound TP4
single-session recipe on 4x B70.

The quality finding is not a full model-accuracy proof; it is a strict local
determinism and behavior gate designed to catch the corruption modes observed
during this optimization campaign. Under that gate, no quality degradation was
detected.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-clonefinal-fulllogits-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T230432Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-clonefinal-fulllogits-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T230432Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T231950Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T232243Z.json`
- LocalMaxxing payload:
  `/home/steve/llm-optimizations-publish/data/localmaxxing-minimax-m27-autoround-no-clone-clonefinal-fulllogits-promoted-p512n1536-20260517.payload.json`
- LocalMaxxing response:
  `/home/steve/llm-optimizations-publish/data/localmaxxing-responses/minimax-m27-autoround-no-clone-clonefinal-fulllogits-promoted-p512n1536-20260517.response.json`

## Follow-Up

Next optimization target: identify whether the remaining output-side overhead
can be reduced without reintroducing quality drift. Good candidates are a
smaller final-logits synchronization boundary, graph-capture cleanup around the
full-logits path, and focused timing around final hidden clone cost versus
output projection and sampler collectives.
