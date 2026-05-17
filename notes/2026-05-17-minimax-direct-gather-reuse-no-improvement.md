# MiniMax Direct-Gather Reuse Candidate

Date: 2026-05-17

## Result

The direct-gather reuse candidate is quality-safe but not faster than the
current promoted MiniMax M2.7 AutoRound baseline.

- Candidate marker: `logits.local_argmax_pair_direct_gather_reuse`
- Runtime flag: `VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER_REUSE=1`
- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Shape: p512 / n1536 / batch 1 / context 2048

Throughput:

| Run | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| repeat 1 | 61.344752 | 81.793003 |
| repeat 2 | 61.234242 | 81.645657 |
| mean | 61.289497 | 81.719330 |

Current promoted strict baseline:

- LocalMaxxing: `cmp9xpe3w04pdo4013acdikt7`
- Mean output tok/s: `61.404035`
- Mean total tok/s: `81.872046`

Decision: do not promote and do not submit to LocalMaxxing. This is a valid
negative result.

## Quality

The candidate passed the full current quality gate:

- raw145 n64 exact token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact token hash:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS, exact arithmetic `42`, and `add_one`
- arithmetic repeat: 16/16 exact `42`
- extended sixpack: PASS, arithmetic, code, JSON, sort, SQL

Summary:

`/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-direct-gather-reuse-tightquality-strict-tp4-ctx2048-mbt512-bs256-20260517T160531Z-summary.json`

## Collective Probe

Standalone XCCL probe:

`benchmarks/b70_xccl_pair_collective_probe.py`

Findings:

- `all_gather_into_tensor` is correct and measures about `0.098` to `0.117`
  ms on rank 0 for a single `(value, token)` pair.
- `all_gather_list` is slower, about `0.23` to `0.24` ms, and mismatched the
  random case.
- `gather_broadcast` timed out.
- `all_to_all_repeated` timed out.

This means the raw XCCL tiny all-gather is not the main bottleneck. The vLLM
rank-0 timing bucket around `logits.local_argmax_pair_all_gather` remains much
larger than the standalone collective, so the remaining gap is likely framework,
graph replay, stream, or per-token orchestration overhead rather than payload
transfer bandwidth.

## Patch Surface

The candidate is default-off and leaves the promoted path untouched.

Runtime/source files touched:

- `/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py`
- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`

Repository support files touched:

- `benchmarks/b70_xccl_pair_collective_probe.py`
- `scripts/inspect-vllm-runtime.py`
- `scripts/run-minimax-strict-quality-gated-candidate.sh`
- `data/minimax-m27-direct-gather-reuse-no-improvement-20260517.json`

## Next

The next serious path should avoid another wrapper around the same gather. More
promising work:

- move more of final token selection and handoff into a graph-resident or
  GPU-resident path;
- inspect why the model path reports milliseconds around a collective that the
  standalone probe measures in tenths of a millisecond;
- revisit EP4 only if we can first make the EP all-to-all and W4A16 expert
  layout quality-correct.
