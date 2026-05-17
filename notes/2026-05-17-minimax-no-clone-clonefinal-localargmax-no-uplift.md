# MiniMax No-Clone + Final-Hidden Clone + Local-Argmax Screen

Date: 2026-05-17

## Summary

This candidate combined three runtime switches:

- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- `VLLM_MINIMAX_M2_CLONE_FINAL_HIDDEN=1`
- `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`

The goal was to keep the quality-stable local-argmax decode path while removing
the compiled all-reduce clone cost. The final hidden-state clone was added as a
guard because the full-logits no-clone path showed deterministic token drift at
the raw145 n256 exact gate.

## Quality Result

The combined candidate passed the full strict gate:

- raw145 n64 exact:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS
- arithmetic repeat: exact `42`, 16 greedy repeats, deterministic
- extended sixpack: PASS

This means the local-argmax branch can tolerate the no-clone all-reduce path
when final hidden state is cloned before logits. That is useful as a correctness
boundary, even though it did not improve speed.

## Benchmark Result

Shape:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Prompt/output/context: p512, n1536, ctx2048
- Batch/concurrency: 1
- Block size: 256
- Prefix cache: disabled
- Temperature: greedy / 0

Two repeats after the full strict quality gate:

| repeat | output tok/s | total tok/s |
| --- | ---: | ---: |
| 1 | 61.121444 | 81.495259 |
| 2 | 60.974112 | 81.298817 |
| mean | 61.047778 | 81.397038 |

Current strict baseline from `2026-05-17-minimax-localargmax-tightquality.md`:

| result | output tok/s | total tok/s |
| --- | ---: | ---: |
| strict baseline mean | 61.404035 | 81.872046 |
| this candidate mean | 61.047778 | 81.397038 |
| delta | -0.356257 | -0.475008 |

## Decision

Reject as a performance promotion. The result is quality-safe, but slower than
the current strict baseline by about `0.58%` decode throughput.

Do not submit this as a new LocalMaxxing achievement row. It is a useful
negative result for the optimization log, because it shows that clone removal
alone is not the path to the next MiniMax speed tier once local-argmax is already
active.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-clonefinal-localargmax-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T215328Z-summary.json`
- Quality directory:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-no-clone-clonefinal-localargmax-bench-strict-tp4-ctx2048-mbt512-bs256-20260517T215328Z-quality`
- Benchmark JSONs:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T220531Z.json`
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T220826Z.json`

## Follow-Up

The next software-side target should move away from clone-only changes and back
toward communication scheduling:

- test XCCL topology override `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`
  under strict gates, because oneCCL reports PCIe-only topology and explicitly
  offers this override when fabric recognition is wrong;
- measure whether the override changes all-reduce latency or decode throughput
  without changing quality;
- continue treating any >60 tok/s result as provisional until it survives exact
  hashes, semantic checks, arithmetic repeat, and extended sixpack.
