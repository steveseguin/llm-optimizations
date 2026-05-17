# MiniMax Packed-Gather And Pair-Reuse Candidates

Date: 2026-05-17

## Result

Two local-argmax communication variants were tested for MiniMax M2.7 AutoRound
INT4 on 4x Intel Arc Pro B70. Both are useful negative results, but neither
should be promoted.

Current promoted strict baseline:

- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`
- Output tok/s: `61.404035`
- Total tok/s: `81.872046`
- Shape: p512 / n1536 / batch 1 / context 2048 / TP4
- Quality: raw145 n64/n256 exact token hashes, semantic suite, arithmetic
  repeat, extended sixpack

## Packed Gather

Runtime flag:

```bash
export VLLM_XPU_LOCAL_ARGMAX_PACKED_GATHER=1
```

This packs the local max float bits plus token tie key into one `int64`,
uses `all_gather_into_tensor`, then reduces locally. It avoids XCCL
`ReduceOp.MAX`, which had already failed quality in previous experiments.

Quality:

- full strict gate passed
- raw145 n64 exact
- raw145 n256 exact
- semantic suite passed
- arithmetic repeat passed
- extended sixpack passed

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| repeat 1 | 58.146237 | 77.528316 |
| repeat 2 | 57.194290 | 76.259053 |
| mean | 57.670263 | 76.893684 |

Decision: reject for speed. It is quality-safe but slower than the promoted
baseline. A no-debug rerun was worse at `26.695537` output tok/s and the next
repeat stalled in shared-memory broadcast, so this is not a stable promotion
path.

## Pair Reuse

Runtime flag:

```bash
export VLLM_XPU_LOCAL_ARGMAX_PAIR_REUSE=1
```

This keeps the promoted float32 `(value, global_token)` pair gather semantics,
but reuses the local pair and gathered buffers instead of allocating them each
decode token.

Quality screen:

- raw145 n64 exact token hash passed
- expected hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| repeat 1 | 45.498382 | 60.664510 |

Decision: reject before full strict gate. It passed the first exact canary, but
the first p512/n1536 benchmark was far below baseline. The second repeat was
stopped to avoid wasting time and system RAM.

## Artifacts

Result data:

- `data/minimax-m27-packed-gather-and-pair-reuse-negative-20260517.json`

Packed-gather artifacts:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/packed-gather-v1/minimax-packed-gather-v1-strict-tp4-ctx2048-mbt512-bs256-20260517T174005Z-summary.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/packed-gather-v1/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T175227Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/packed-gather-v1/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T175523Z.json`

Pair-reuse artifacts:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/pair-reuse-screen/raw145-n64-pair-reuse-20260517T181226Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/pair-reuse-v1-bench/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T181909Z.json`

## Lesson

Reducing allocation around the existing pair gather did not help, and packing
the pair into a single `int64` made the path slower. The bottleneck is unlikely
to be just tensor allocation or the raw payload size. The next useful work is a
larger change: keep token selection and sampler handoff graph/GPU resident, or
write a deterministic XPU local-argmax primitive that avoids the current
framework path around the collective.

No LocalMaxxing submission was made for these candidates because neither
improved the promoted result.
