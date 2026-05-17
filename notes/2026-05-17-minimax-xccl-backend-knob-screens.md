# MiniMax XCCL Backend Knob Screens

Date: 2026-05-17

## Summary

These screens did not produce a new MiniMax promotion. They narrow the bottleneck
around the final local-argmax TP exchange.

Current promoted strict baseline remains:

- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`
- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Shape: p512 / n1536 / batch 1 / context 2048 / TP4
- Mean output tok/s: `61.404035`
- Mean total tok/s: `81.872046`

No LocalMaxxing submission was made for these screens.

## Rank Skew

All-rank timing on the promoted path shows the pair all-gather bucket is nearly
the same on every rank:

| Rank | Pair all-gather avg ms |
| ---: | ---: |
| 0 | 7.986459 |
| 1 | 7.898390 |
| 2 | 7.950471 |
| 3 | 8.016030 |

Throughput for this short p512/n512 profile was `60.395565` output tok/s and
`120.791130` total tok/s.

Interpretation: this is not a simple one-rank-late problem. The collective
bucket itself is consistently expensive in the vLLM runtime context.

Artifacts:

- `/home/steve/bench-results/minimax-m2.7-localargmax-rankskew-profile/vllm-minimax-m27-autoround-tp4-p512n512-20260517T163923Z.log`
- `/home/steve/bench-results/minimax-m2.7-localargmax-rankskew-profile/vllm-minimax-m27-autoround-tp4-p512n512-20260517T163923Z.json`

## Direct Gather Reuse

The default-off `VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER_REUSE=1` branch did not
improve speed.

| Rank | Direct gather-reuse avg ms |
| ---: | ---: |
| 0 | 7.680680 |
| 1 | 8.162359 |
| 2 | 7.753192 |
| 3 | 7.821414 |

Short p512/n512 throughput was `59.664325` output tok/s and `119.328649` total
tok/s, slower than the promoted path.

## List All-Gather

`VLLM_XPU_LIST_ALL_GATHER=1` passed the raw145 n64 exact-token screen:

- token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- elapsed for quality screen: `10.868426` s

The p512/n256 throughput screen did not show an improvement:

- output tok/s: `51.961538`
- total tok/s: `155.884614`
- pair all-gather still measured about `7.83` to `8.03` ms by rank

Decision: do not promote, and do not run the full strict gate.

## Fabric Vertex Check Override

`CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` also passed raw145 n64 exact tokens,
but it was far slower:

- token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- elapsed for quality screen: `26.753757` s
- rough output rate during that screen: `2.392187` tok/s

Decision: do not benchmark further.

## Takeaway

The raw standalone XCCL tiny all-gather probe can measure around `0.1` ms, but
the vLLM MiniMax decode path still spends about `8` ms per token in the final
TP local-argmax exchange. Reusing the output buffer, switching to list
all-gather, and forcing CCL fabric assumptions did not remove that cost.

The next serious path should be a different design, not another wrapper around
the same collective:

- custom XPU/Level Zero pair reduction for `(float32 value, int32 token)`;
- GPU-resident token handoff to reduce framework/CPU involvement;
- exact speculative decoding only where verifier-equivalent quality and honest
  acceptance rates can be shown.
