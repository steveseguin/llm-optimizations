# MiniMax Native Router And c158 Cache Screens, 2026-05-12

## Goal

Continue the MiniMax M2.7 AutoRound INT4 four-B70 optimization work after the
CCL/AOT recheck. The specific questions were:

- can llm-scaler avoid more MiniMax routing overhead without changing selected
  experts or route weights;
- can the old `c158...` compiled cache explain the prior `41.130667` output
  tok/s p512/n1536 result;
- which patches are worth keeping as future starting points.

All throughput runs below used 4x Arc Pro B70, MiniMax M2.7 AutoRound W4A16,
vLLM/XPU TP4, FP16 activations, llm-scaler u4 MoE enabled, XPU graph disabled,
no speculation, no expert dropping, and no power-limit changes.

## Native MiniMax Logits Path

I added a default-off MiniMax-specific extension path:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS=1
```

Shape:

- input `router_logits [T,256]`;
- exact MiniMax selection score `sigmoid(router_logits) + e_score_bias`;
- route weight source `sigmoid(router_logits)`;
- top-k `8`, renormalized;
- output feeds the existing unsigned-u4 tiny MoE implementation.

Synthetic XPU equivalence against the existing exact PyTorch top-k path:

```text
max_abs_diff=0.0
mean_abs_diff=0.0
```

Results:

| Run | Shape | Cache/AOT | KV tokens | Total tok/s | Output tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MiniMax logits smoke | p64/n16 | cold `ca72...` | n/a | `17.15` | `3.43` | cold smoke only |
| MiniMax logits warm smoke | p64/n16 | AOT `ca72...` | `18,432` | `99.93` | `19.99` | healthy but not decisive |
| MiniMax logits stale-cache screen | p512/n512 | reused current `3b096...` | n/a | `74.03` | `37.01` | not clean; stale AOT caveat |
| MiniMax logits isolated cold | p512/n512 | compiled `3b096...` | `9,408` | `56.903210` | `28.451605` | cold artifact |
| MiniMax logits isolated warm | p512/n512 | loaded `3b096...` | `17,216` | `75.896741` | `37.948370` | negative vs `39.610585` p512/n512 reference |

Decision: do not promote. The path is quality-preserving in the synthetic
check, but the extra native top-k wrapper does not beat vLLM's current compiled
router plus u4 MoE path at the real p512/n512 screen.

Patch artifacts:

- `patches/llm-scaler-minimax-router-and-u4-experiments-20260512.patch`
- `patches/vllm-minimax-router-and-moe-experiments-20260512.patch`

## u4 Work-Sharing Variant

I also added:

```bash
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
```

This routes the default selected experts through the llm-scaler work-sharing
unsigned-u4 tiny MoE variant.

Result:

| Run | Shape | Total tok/s | Output tok/s | Decision |
| --- | --- | ---: | ---: | --- |
| u4 work-sharing | p512/n512 | about `68.37` | about `34.18` | negative |

Decision: leave the flag unset for real runs.

## Native Candidate Repair

The existing vLLM MiniMax candidate-router hook proposes candidate experts with
an FP16 gate, then repairs routing by recomputing exact FP32 scores over those
candidates. Earlier Python/torch repair was too slow and the standalone SYCL
router module had device-image registration problems.

I moved the standalone repair op into the already loadable `moe_int4_ops`
extension and registered:

```text
torch.ops.moe_int4_ops.minimax_m2_candidate_repair_topk
```

vLLM can enable it with:

```bash
VLLM_MINIMAX_M2_CANDIDATE_ROUTER_TOPM=16
VLLM_MINIMAX_M2_CANDIDATE_ROUTER_XPU_REPAIR=1
```

Synthetic XPU correctness:

```text
op_registered=True
ids_equal=True
max_abs_diff=2.9802322387695312e-08
mean_abs_diff=1.257285475730896e-08
```

Results:

| Run | Shape | Candidate count | Cache/AOT | KV tokens | Total tok/s | Output tok/s | Decision |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| repair smoke cold | p64/n16 | 16 | cold `6a952...` | `7,936` | `16.490200` | `3.298040` | cold smoke only |
| repair smoke warm | p64/n16 | 16 | loaded `6a952...` | `16,832` | `100.995073` | `20.199015` | healthy |
| repair cold | p512/n512 | 16 | compiled `3b096...` | `7,808` | `56.241644` | `28.120822` | cold artifact |
| repair warm | p512/n512 | 16 | loaded `3b096...` | `15,680` | `72.648622` | `36.324311` | negative |
| repair top12 cold | p512/n512 | 12 | compiled `3b096...` | `7,808` | `55.665562` | `27.832781` | cold artifact |
| repair top12 warm | p512/n512 | 12 | loaded `3b096...` | `15,680` | `71.884592` | `35.942296` | negative; also less quality margin than top16 |

Decision: native repair is correctly registered and numerically matches the
exact candidate repair calculation, but it is slower than the current
quality-cleared p512/n512 floor. The likely issue is that FP16 proposal top-k
plus exact candidate-dot repair adds more scheduling and scalar work than it
removes. Keep as a patch artifact only.

## c158 Cache Recheck

The old fast archive remains:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-c158-archive-20260510T070154Z
old AOT=c15860ddb8a1077c5ba1a1ae2d0f86552a357eb56772cdbf02828195b5a363ec
old cache key=1d97049441
```

Running the current stack with that cache root did not load c158. vLLM computed
the current cache key and wrote a new `3b096...` AOT inside the archive:

| Run | Shape | What happened | KV tokens | Total tok/s | Output tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| archive root recheck | p512/n1536 | compiled current `3b096...` into archive root | `9,408` | `44.578611` | `33.433958` |

I then created a separate diagnostic cache root that copied the old c158 models
under the current expected `3b096...` AOT path:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-c158-forced-3b096-20260512T071035Z
```

vLLM refused to load it:

```text
Compiling model again due to a load failure ... reason: Source code has changed since the last compilation.
```

It recompiled current code and overwrote the forced model files:

| Run | Shape | What happened | KV tokens | Total tok/s | Output tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| forced c158 diagnostic | p512/n1536 | rejected old binary, recompiled current `3b096...` | `17,216` | `49.579031` | `37.184273` |

Decision: the old c158 speed cannot be reproduced under the current source
guard. Treat `41.130667` only as a scheduling clue. vLLM's AOT source-change
guard is doing the right thing and prevents accidentally promoting a stale
compiled graph.

## Conclusions

- Current valid non-spec MiniMax baseline remains `38.046755` output tok/s at
  p512/n1536 and `39.610585` output tok/s at p512/n512.
- The native logits path is exact but not faster enough to promote.
- Native candidate repair works and is useful as a code artifact, but it is
  slower in real p512/n512 decode.
- The u4 work-sharing path is negative.
- The c158 cache lead is closed as non-reproducible with current source; vLLM
  recompiles it due source-change detection.

## Next Work

The next work should move away from router micro-fusion and back to graph
boundaries that actually dominate decode:

- XPU-specific Q/K variance allreduce plus RMS application;
- hidden-state allreduce plus adjacent residual/RMS or MoE epilogue work;
- AOT/generated-cache comparison focused on the current `3b096...` graph, not
  stale c158 binaries;
- lower-overhead attention/KV timing that does not perturb vLLM shared-memory
  startup or decode scheduling.

Structured data:

`data/minimax-m27-native-router-and-c158-screens-20260512.json`
