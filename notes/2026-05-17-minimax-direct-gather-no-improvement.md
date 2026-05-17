# MiniMax Direct Gather Local-Argmax Check

Date: 2026-05-17

## Result

The direct `dist.all_gather_into_tensor` local-argmax path is quality-valid on
the current 4x B70 MiniMax M2.7 AutoRound stack, but it did not improve
throughput over the promoted pair all-gather baseline.

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4
- Quantization: AutoRound INT4 W4A16
- Candidate marker: `logits.local_argmax_pair_direct_gather`
- Shape: p512 / n1536 / batch 1 / context 2048
- Output throughput: `61.086391` tok/s
- Total throughput: `81.448522` tok/s
- Current promoted baseline: `61.317497` output tok/s, `81.756663` total tok/s

No LocalMaxxing submission was made because this is a valid no-improvement
candidate, not a new performance result.

## Quality Gates

The candidate passed the same strict quality screen used for promoted results:

- raw145 n64 exact combined token hash:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact combined token hash:
  `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: PASS prompt, arithmetic `42`, and valid `add_one` function
- arithmetic repeat suite: 8 greedy repeats, deterministic, all matched `42`

## Interpretation

The result narrows the logits-stage search:

- XCCL packed `ReduceOp.MAX` remains rejected because the standalone probe shows
  incorrect reductions.
- `all_gather_into_tensor` is correctness-safe here, but it is not faster than
  the existing promoted pair all-gather route.
- The next meaningful speed path is not another wrapper around the same gather.
  It should be a standalone-probed custom pair reducer or Level Zero peer
  reduction that compares `(float32 value, int32 token)` directly and preserves
  exact greedy-token behavior.

## Artifacts

- Summary:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-direct-gather-qualityscreen-strict-tp4-ctx2048-mbt512-bs256-20260517T123251Z-summary.json`
- Benchmark JSON:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T124540Z.json`
- Data summary:
  `data/minimax-m27-direct-gather-no-improvement-20260517.json`
