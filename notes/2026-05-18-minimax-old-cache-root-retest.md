# MiniMax M2.7 Old Cache Root Retest

UTC timestamp: 2026-05-18T02:07:01Z

## Goal

Retest the older May 13 XPU graph cache root that previously produced a 73.306 output tok/s single run, but under the current strict quality-gated harness. The purpose was to determine whether the older cache/runtime state could explain the gap between the current strict baseline near 70 tok/s and the older high result.

## Configuration

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM/XPU local build, tensor parallel 4
- Quantization: AutoRound INT4 W4A16
- Runtime: float16, FlashAttention/default backend, XPU graph enabled, communication ops forced into graph
- Benchmark shape: prompt 512, output 1536, context 2048, batch 1
- Cache root: `/mnt/fast-ai/vllm-cache-exp/minimax-xpugraph-attndelay-block256-mbt512-noprefix-20260513T171301Z`

## Quality Result

Status: `quality_passed`

Strict gates passed before benchmarking:

- `raw145-n64-exact`: hash `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- `raw145-n256-exact`: hash `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r16`
- `extended-sixpack-n64-r2`

## Performance

Two strict post-quality benchmark repeats:

| Repeat | Output tok/s | Total tok/s | Elapsed s |
| --- | ---: | ---: | ---: |
| 1 | 70.1749 | 93.5666 | 21.8882 |
| 2 | 69.8192 | 93.0923 | 21.9997 |

Mean output tok/s: `69.9971`

Mean total tok/s: `93.3294`

## Findings

This did not recover the earlier 73.306 tok/s result. Despite pointing at the old cache root, the runtime loaded the current AOT artifact:

`03f6a28c070656d44eab4c581bc8dc5295ed123e7c0150c7f596ea24012406b0`

It did not load the older `d0fed...` AOT artifact. The result is therefore best interpreted as a cache-root/runtime-state diagnostic, not a true old-AOT replay.

Each benchmark process also showed very heavy host IO and major page faults during process lifetime because the 112 GB checkpoint is much larger than available system RAM:

| Repeat | Major page faults | File-system inputs |
| --- | ---: | ---: |
| 1 | 1,002,769 | 248,822,136 |
| 2 | 1,015,635 | 248,746,624 |

That mainly affects model load and repeat turnaround, but it is a real reproducibility concern while RAM is constrained.

## Conclusion

The current honest strict plateau remains approximately 70 output tok/s for p512/n1536 on 4x B70. The older 73.306 tok/s result should remain recorded as an interesting historical high, but not promoted over the stricter 70.006-69.997 range unless the old AOT can be recreated and passes the same gates.

Next work should target real communication-boundary reduction or epilogue fusion rather than more cache-root retesting.
