# MiniMax Gather-Broadcast Erratum - 2026-05-17

This corrects the earlier gather/broadcast local-argmax note and LocalMaxxing
submission.

## What Was Wrong

The files added in `Record MiniMax gather-broadcast argmax follow-up` described
the p512/n1536 result as a gather/broadcast argmax benchmark. Follow-up runtime
inspection showed that was not true:

- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py` had the
  new `VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST` branch.
- The active benchmark runtime imported
  `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`.
- That installed site-packages copy did not contain the gather/broadcast branch
  when the first benchmark and LocalMaxxing submission were run.

The timing rerun made the mismatch visible because, even with
`VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST=1`, the runtime emitted the old labels:

- `logits.local_argmax_pair_all_gather`
- `logits.local_argmax_reduce`
- `logits.local_argmax_pair_stack`
- `logits.local_argmax_local_max`

Artifact:

- `/home/steve/bench-results/minimax-m2.7-low-overhead-timing/vllm-minimax-m27-autoround-tp4-p512n512-20260517T092048Z.log`

## LocalMaxxing Impact

The submitted row is:

- `cmp9jyd8m049ko401kk2n1pju`

Treat that row as a valid strict-quality MiniMax-logits local-argmax
pair-all-gather run, but not as validation of gather/broadcast. Its metrics were:

- Mean output: 61.475994 tok/s
- Mean total: 81.967992 tok/s
- Shape: p512/n1536, batch 1, 4x B70, AutoRound W4A16

An authenticated `PATCH /api/benchmarks/cmp9jyd8m049ko401kk2n1pju` attempt to
correct the notes returned HTTP 404. The API documentation available in this
session does not expose an update/delete benchmark endpoint, so the correction
is recorded here rather than edited in place.

## Real Gather-Broadcast Test

After identifying the runtime mismatch, the active venv copy was patched with
the gather/broadcast branch and verified:

```text
/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py
has_gather_broadcast= True
```

Then the real gather/broadcast path was tested with the strict raw145 n64 gate:

- Label: `minimaxlogits-localargmax-real-gatherbroadcast-qualityonly`
- Summary stem:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-real-gatherbroadcast-qualityonly-strict-tp4-ctx2048-mbt512-bs256-20260517T092752Z`
- Log:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-real-gatherbroadcast-qualityonly-strict-tp4-ctx2048-mbt512-bs256-20260517T092752Z-quality/raw145-n64-exact.log`
- JSON: none emitted.

Result:

- The engine loaded, compiled, captured graphs, rendered the prompt, and entered
  first decode.
- It then stalled at `Processed prompts: 0/1`.
- It emitted repeated `No available shared memory broadcast block found in 60 seconds`
  warnings after graph capture.
- The run was manually terminated after the second post-decode stall warning.

Decision: real `VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST=1` is rejected for the
current graph path. It is not quality-safe because it does not complete the
first exact-token gate.

## Corrected Interpretation

- The current promoted strict result remains the MiniMax-logits local-argmax
  pair-all-gather path, not gather/broadcast.
- The apparent gather/broadcast 61.476 tok/s row is a duplicate/noise-level
  repeat of the default pair-all-gather path.
- Gather/broadcast appears to introduce a graph/collective ordering deadlock
  around first decode.
- Future benchmark candidates must verify the active venv import path before
  running, not only the source tree under `/home/steve/src/vllm`.

## Next Action

Do not spend more time on root gather/broadcast unless debugging collective
ordering inside XPU graph capture specifically. The better next path remains:

- reduce CPU/framework callbacks without adding new graph-visible collectives;
- investigate a GPU-resident top-token handoff that does not require a separate
  gather+broadcast sequence;
- repair EP4/MoE layout if pursuing a larger decode gain;
- keep strict raw145 and semantic canaries as the gate before speed claims.