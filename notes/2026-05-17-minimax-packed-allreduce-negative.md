# MiniMax Packed Local-Argmax AllReduce Negative

Date: 2026-05-17

## Result

The packed one-collective local-argmax path is not quality-safe in the current
vLLM/XPU runtime.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Candidate flag: `VLLM_XPU_LOCAL_ARGMAX_PACKED_ALLREDUCE=1`
- Runtime guard: required `logits.local_argmax_packed_allreduce`
- Outcome: failed before benchmarking

The candidate passed runtime import verification and reached normal model load,
compile, KV allocation, and graph capture. It failed the first raw145 n64 exact
token-hash quality gate:

- expected token hash: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- candidate token hash: `ade0b2e05944a4964b3dd0ba0bfa90fa8a5341bc3e3f2ea885e413e0e283291f`
- failure reasons: combined token hash mismatch, degenerate/corrupt generated output
- generated text had zero printable non-space characters
- generated tokens were deterministic but wrong

No throughput benchmark was run and no LocalMaxxing submission was made.

## Interpretation

The packed key path is attractive because it reduces the logits-stage TP
communication from pair all-gather plus local reduce to a single all-reduce.
In practice, this implementation does not preserve greedy argmax semantics on
the B70/XCCL path. The likely issue is in the float-to-sort-key/int64 packing
or the signed integer all-reduce ordering, not model math before the logits
stage.

Keep the runtime-guarded pair all-gather path as the promoted quality-preserving
baseline for now:

- LocalMaxxing: `cmp9q9fzn04cto401tjcila06`
- mean output throughput: `61.317497` tok/s
- mean total throughput: `81.756663` tok/s

## Next Leads

- If revisiting this, build a tiny standalone XPU/XCCL reproducer for the
  packed float32 key ordering before touching the model path again.
- Prefer a custom XPU reduction op that compares `(float32 value, int32 token)`
  directly instead of packing into signed int64 keys.
- Continue toward GPU-resident token handoff or an XPU custom argmax collective,
  but require the same raw145 and semantic quality gates before benchmarking.
