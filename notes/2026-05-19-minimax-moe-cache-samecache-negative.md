# MiniMax MoE Callable Cache Same-Cache A/B Negative

Date: 2026-05-19

## Goal

Recheck the MiniMax llm-scaler W4A16 MoE callable-cache candidate with a cleaner
same-cache counterfactual. The earlier full run with
`VLLM_XPU_LLM_SCALER_MOE_CACHE_MINIMAX_LOGITS_OP=1` was quality-clean and
measured slightly above the clean promoted baseline, but the no-cache
counterfactual was unstable enough that the result stayed unpromoted.

This follow-up used the same cache root for both sides:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-moe-cache-ab-20260519
```

Both sides used the clean direct Q/K variance in-place scale recipe and the
same strict canaries before benchmark repeats.

## Cache-On Result

Label:

```text
minimax-moe-cache-ab-cacheon-20260519
```

Quality passed:

- `raw145-n64-exact`
- `raw145-n256-exact`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r8`

Benchmark:

```text
output tok/s: [88.6123416469214, 88.71182949768006, 87.98204832438597, 88.31259261112109]
total tok/s:  [118.14978886256186, 118.28243933024008, 117.30939776584798, 117.75012348149478]
mean output: 88.40470302002713
mean total:  117.87293736003618
```

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-cache-ab-cacheon-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T113341Z-summary.json
```

## Cache-Off Counterfactual

Label:

```text
minimax-moe-cache-ab-cacheoff-20260519
```

Quality passed:

- `raw145-n64-exact`
- `raw145-n256-exact`
- `semantic-suite-n64-r2`
- `arithmetic-repeat-n64-r8`

Benchmark:

```text
output tok/s: [88.06446552619066, 88.04887043639482]
total tok/s:  [117.41928736825422, 117.3984939151931]
mean output: 88.05666798129275
mean total:  117.40889064172366
```

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-moe-cache-ab-cacheoff-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T115822Z-summary.json
```

## Decision

Reject and do not submit to LocalMaxxing.

The callable cache is quality-safe in this screen and shows a small same-cache
advantage over cache-off, but the cache-on mean `88.404703` output tok/s is
below the current clean promoted `88.501953` output tok/s baseline. This is not
a real improvement toward the >60 tok/s quality-preserving goal because that
goal has already been exceeded; current promotion requires beating the clean
88.5 tok/s path repeatably.

If this patch is kept, treat it as optional code cleanup for later MoE boundary
work, not as a promoted speed path.

## Artifacts

- Data: `data/minimax-m27-moe-cache-samecache-negative-20260519.json`
- Prior marginal result: `notes/2026-05-19-minimax-moe-cache-minimax-op-marginal.md`
