# B70 XCCL Packed Argmax Probe

Date: 2026-05-17

## Result

A standalone XPU/XCCL probe reproduced the packed local-argmax failure outside
the MiniMax model.

- Script: `benchmarks/b70_xccl_packed_argmax_probe.py`
- Command shape: `python -m torch.distributed.run --nproc_per_node=4 ...`
- Hardware: 4x Intel Arc Pro B70 32GB
- Backend: PyTorch XPU distributed backend `xccl`

The packed key path failed every case on every rank:

- `rank_wins`: 8/8 mismatches on each rank
- `mixed_sign`: 8/8 mismatches on each rank
- `negative_only`: 8/8 mismatches on each rank
- `token_tie`: 8/8 mismatches on each rank
- `random`: 8/8 mismatches on each rank

Example from `rank_wins`:

- reference first tokens: `[300000, 300001, 300002, 300003]`
- packed candidate first tokens: `[600003, 600007, 600011, 600015]`

That candidate pattern is consistent with summed low token/tie fields rather
than a max-selected packed key.

## ReduceOp Smoke

The same probe also runs plain all-reduce smoke checks. On this stack,
`ReduceOp.MAX` did not produce the expected max value for XPU `int32`, `int64`,
or `float32` tensors in the probe. Because the SUM smoke also showed suspicious
state interaction for some dtype/op sequences, the safest current conclusion is
not a narrow "int64 MAX only" bug, but:

`torch.distributed` XCCL reduction ops other than the proven model path should
not be assumed safe without a standalone correctness probe on B70.

## Optimization Consequence

Do not promote the packed `int64` all-reduce argmax branch. The next logits-stage
speed work should avoid packed-key XCCL `MAX` and instead use one of:

- a custom XPU op that directly compares `(float32 value, int32 token)` pairs
- a Level Zero peer-memory/mailbox reduction
- the existing quality-promoted pair all-gather path until a custom reducer passes
  the raw145 and semantic gates

No LocalMaxxing submission was made for this diagnostic because it is not a
throughput result.
