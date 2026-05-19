# MiniMax Current-High Block Size 128 Quality Fail

Date: 2026-05-19

## Summary

This run retested `--block-size 128` on top of the current strict MiniMax
high-speed recipe. Earlier block-size 128 data came from older recipes, so this
screen checked whether the smaller KV block size was still viable after the
current MoE custom-op and allreduce work.

The only intended configuration-level change versus the promoted recipe was:

```bash
BLOCK_SIZE=128
```

The promoted recipe keeps `--block-size 256`.

## Quality

The candidate failed the first strict exact-token gate:

- Gate: `raw145-n64-exact`
- Expected combined token SHA256:
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- Observed combined token SHA256:
  `7f4222437a76869f7fde3e202d3c90a5e202583f603b57d4c9950fae6ad8bd67`
- Deterministic across runs: `true`
- Failure reason: `combined token hash mismatch`

The output was non-degenerate and did not show control-token corruption, but it
was not exact-token equivalent to the promoted reference. Under the current
quality policy, that is a hard reject.

## Performance

No throughput benchmark was run. The strict harness stopped before benchmarking
because `raw145-n64-exact` failed.

## Decision

Reject. Do not submit to LocalMaxxing. Keep `--block-size 256` for the current
promoted MiniMax recipe.

The useful lesson is that KV/block scheduling can change generated tokens even
without changing weights, quantization, router logic, sampling, or power
settings. Block-size sweeps must remain behind exact-token quality gates.

## Artifacts

- Strict summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-currenthigh-block128-20260519-strict-tp4-ctx2048-mbt512-bs128-20260519T173713Z-summary.json`
- Failed quality JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-currenthigh-block128-20260519-strict-tp4-ctx2048-mbt512-bs128-20260519T173713Z-quality/raw145-n64-exact.json`
- Failed quality log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-currenthigh-block128-20260519-strict-tp4-ctx2048-mbt512-bs128-20260519T173713Z-quality/raw145-n64-exact.log`
- Local data: `data/minimax-m27-currenthigh-block128-quality-fail-20260519.json`
