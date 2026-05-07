# 2026-05-07 Q4 Scheduler And Layout Screens

## Summary

This pass closed three current-stack Q4_0 single-card branches on one B70 (`SYCL2`, `-sm none`, `-fa 1`, `-ub 128`, `f16` KV):

- exact-layout subgroup versus ESIMD: the active subgroup reordered MMVQ is faster than the ESIMD blockscale harness when both use llama.cpp-like reordered layout;
- non-adjacent same-shape MMVQ2 lookahead: safe in smoke, but speed-neutral;
- mixed-row `attn_qkv + attn_gate` fusion with deferred copy: removes launches, but decode speed is effectively flat.

None of these should be promoted into the default recipe yet.

## Exact-Layout Kernel Screen

The standalone ESIMD harness now has a `--kernel subgroup|esimd` mode. With the exact subgroup layout used by llama.cpp's reordered Q4_0 MMVQ, subgroup is faster across the tested Qwen3.6 shapes:

| shape / mode | subgroup us | ESIMD blockscale us |
| --- | ---: | ---: |
| `17408 x 5120` single | `85.417` | `99.687` |
| `17408 x 5120` fused2 | `169.583` | `177.708` |
| `4352 x 5120` single | `11.562` | `27.604` |
| `4352 x 5120` fused2 | `24.896` | `44.375` |
| `5120 x 17408` single | `85.417` | `107.708` |
| `5120 x 17408` fused2 | `174.375` | `200.938` |
| `5120 x 1536` single | `5.417` | `12.187` |
| `5120 x 1536` fused2 | `7.500` | `17.291` |

Decision: do not port the ESIMD blockscale branch directly into llama.cpp yet. The next useful kernel work is either a more substantial ESIMD redesign or more graph-level launch/schedule reduction.

## MMVQ2 Lookahead

Runtime gate:

```bash
GGML_SYCL_FUSE_MMVQ2_LOOKAHEAD=1
GGML_SYCL_FUSE_MMVQ2_LOOKAHEAD_MAX=24
```

One-token debug did not change the visible kernel mix versus the baseline:

- plain Q4 reordered MMVQ: `528`;
- fused2: `16`;
- fused2+SwiGLU: `64`;
- RMS_NORM+MUL: `418`.

No-debug A/B at `p0/n128/r2`:

| mode | tok/s |
| --- | ---: |
| off | `24.930745` |
| on | `24.930451` |

Decision: keep disabled. The lookahead path is useful as scaffolding, but it does not remove additional launches in the current graph.

## Mixed `attn_qkv + attn_gate`

Runtime gate:

```bash
GGML_SYCL_FUSE_MMVQ2_MIXED=1
GGML_SYCL_FUSE_MMVQ2_LOOKAHEAD_MAX=80
```

The correct version must use a deferred temporary for `z`, because the graph allocator may reuse `z`'s final destination for intervening linear-attention temporaries. With the deferred path enabled, debug counts changed as intended:

- plain Q4 reordered MMVQ: `528 -> 432`;
- mixed fused2: `0 -> 48`;
- deferred copies: `48`;
- Q8 activation conversions: `676 -> 580`.

No-debug A/B at `p0/n128/r2` with the normal Q8 activation cache:

| mode | tok/s |
| --- | ---: |
| off | `25.088701` |
| on | `25.092403` |

With Q8 cache disabled:

| mode | tok/s |
| --- | ---: |
| off | `24.857660` |
| on | `24.890995` |

Decision: keep disabled. This is a useful diagnostic because it proves the graph has 48 removable `attn_qkv + attn_gate` launches, but the deferred copy cancels the gain. A production version likely needs graph reordering before allocation, or a deeper fused linear-attention block, not a backend-only late scheduler rewrite.

## Artifacts

- Layout TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/layout-compare-20260506/q4-layout-compare-20260506T233237Z.tsv`
- Lookahead A/B TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/lookahead-20260506/lookahead-ab-p0n128-20260506T234825Z.tsv`
- Mixed deferred debug log: `/home/steve/bench-results/qwen36-q4_0-gguf/mixed-fuse-20260506/mixed-qkv-gate-deferred-debug-p0n1-20260507T002117Z.log`
- Mixed deferred A/B TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/mixed-fuse-20260506/mixed-qkv-gate-ab-p0n128-20260507T002210Z.tsv`
- Mixed no-cache A/B TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/mixed-fuse-20260506/mixed-qkv-gate-q8off-ab-p0n128-20260507T002349Z.tsv`
