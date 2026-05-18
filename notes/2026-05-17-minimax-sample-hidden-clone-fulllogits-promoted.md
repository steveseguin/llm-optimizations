# MiniMax Sample-Hidden Clone Full-Logits Promoted Result

Date: 2026-05-17

## Summary

This candidate keeps the quality-safe full-logits path from the previous
promoted run, but moves the XPU safety clone from the whole final hidden state
to the selected sample hidden state immediately before the LM head.

Result: promoted as a small but valid improvement. It passed the full strict
quality harness and averaged `66.609321` output tok/s, compared with
`66.485355` output tok/s for the previous quality-passed promoted result.

LocalMaxxing accepted the result as `cmpag0uvt004io101t6rm25o1`.

This is not a major speed breakthrough. It is useful because it narrows the
correctness barrier and shows that cloning the selected sample hidden state is
enough to preserve the deterministic full-logits path for this benchmark.

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
- `VLLM_XPU_CLONE_SAMPLE_HIDDEN=1`
- `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=0`
- `CCL_TOPO_P2P_ACCESS=1`
- No local-argmax path: `VLLM_XPU_LOCAL_ARGMAX_DECODE` unset

Representative launch:

```bash
LABEL=sample-hidden-clone-fulllogits-bench \
BENCH_REPEATS=2 \
RUN_EXTENDED_QUALITY=1 \
RUN_REPEAT_ARITHMETIC_QUALITY=1 \
REPEAT_ARITHMETIC_RUNS=16 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache-exp/minimax-strict-sample-hidden-clone-fulllogits-20260517 \
VLLM_RUNTIME_REQUIRE_ANY_MARKERS=VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE,VLLM_XPU_CLONE_SAMPLE_HIDDEN \
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1 \
VLLM_XPU_CLONE_SAMPLE_HIDDEN=1 \
VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=0 \
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

Under this corruption-focused harness, no quality degradation was detected.

## Benchmark Result

Two p512/n1536 throughput repeats after quality gates:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 66.593506 | 88.791341 |
| 2 | 66.625136 | 88.833515 |
| mean | 66.609321 | 88.812428 |

Comparison:

- Previous promoted strict full-logits result:
  `66.485355` output tok/s, `88.647140` total tok/s
- Improvement versus previous promoted result:
  `+0.186%` output tok/s
- Same-day quality-safe local-argmax anchor:
  `60.642082` output tok/s, `80.856109` total tok/s
- Improvement versus same-day local-argmax anchor:
  `+9.84%` output tok/s

## Decision

Promote this as the current best quality-safe MiniMax M2.7 AutoRound TP4
single-session recipe on 4x B70. The gain is small, so the practical value is
more about correctness isolation than raw throughput.

## Patch

Patch: `patches/vllm-xpu-clone-sample-hidden-20260517.patch`

This patch assumes the existing local XPU timing/runtime guard work is already
applied, including `os`, `timed_region`, and `_select_sample_hidden_states`.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-sample-hidden-clone-fulllogits-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T234136Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-sample-hidden-clone-fulllogits-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T234136Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T235700Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T235955Z.json`
- LocalMaxxing payload:
  `/home/steve/llm-optimizations-publish/data/localmaxxing-minimax-m27-autoround-sample-hidden-clone-fulllogits-promoted-p512n1536-20260517.payload.json`
- LocalMaxxing response summary:
  `/home/steve/llm-optimizations-publish/data/localmaxxing-responses/minimax-m27-autoround-sample-hidden-clone-fulllogits-promoted-p512n1536-20260517.response.json`

## Follow-Up

The remaining decode gap is not solved by output-tail work alone. The next
useful path is deeper timing inside the compiled model body: MiniMax attention
and MoE scheduling, allreduce boundaries, and whether any final synchronization
can be folded into the LM-head or sampler path without changing output tokens.
