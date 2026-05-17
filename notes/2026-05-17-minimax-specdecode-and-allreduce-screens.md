# MiniMax M2.7 Spec Decode And Allreduce Screens

Date: 2026-05-17

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound` at `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

Current strict baseline remains `61.404035` output tok/s, `81.872046` total tok/s on 4x B70 TP4 with local argmax. LocalMaxxing result: `cmp9xpe3w04pdo4013acdikt7`.

## What Was Tested

### vLLM speculative ngram GPU

Config:

```json
{"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}
```

Outcome:

- Default graph path failed startup in the ngram GPU dummy run:
  `AssertionError: Expected exactly one compiled range_entry for static shape compilation, but found 2`.
- Disabling auto compile ranges got past that assertion, but model load then failed with XPU OOM while trying to allocate another 144 MiB on GPU 2.

Decision: do not promote. On this stack, the GPU ngram proposer adds enough compile/runtime pressure to break the 4x32GB MiniMax setup.

### vLLM speculative ngram CPU

Config:

```json
{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":3,"prompt_lookup_max":8}
```

Outcome:

- vLLM disabled async scheduling for ngram speculative decoding.
- The local-argmax fast path was enabled for initial steps, then disabled once speculative metadata was present.
- The first completed run showed about `4.49` output tok/s from vLLM's progress output, far below baseline.
- A later rerun after a harness JSON fix hit `UR_RESULT_ERROR_OUT_OF_RESOURCES` in Level Zero during compiled attention/KV initialization.

Decision: do not promote. Stock CPU ngram may preserve target-model verification semantics, but it sacrifices the fast path that makes the current result viable.

### Two-allreduce local argmax

Env:

```bash
VLLM_XPU_LOCAL_ARGMAX_ALLREDUCE=1
```

Outcome:

- Exact raw145 n64 quality gate failed.
- Output degenerated to repeated invalid token id `-4`.
- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/localargmax-allreduce-screen/raw145-n64-localargmax-allreduce-20260517T172005Z.json`

Decision: do not promote. XCCL ReduceOp-based argmax paths are unsafe until independently proven correct.

## Harness Update

`scripts/run-vllm-minimax-quality-check.py` now accepts `--speculative-config` and records a JSON-safe copy of the requested config. This allows speculative decode candidates to be tested under the same exact-token gates as other MiniMax changes.

## Next Direction

The most useful optimization target is still the decode-side collective around local argmax. Existing measurements show the pair all-gather bucket costs about 8 ms per generated token on all ranks. Existing XCCL all-reduce variants are not quality-safe, so the next viable options are:

- a deterministic custom XPU pair-reduction path that avoids broken ReduceOp behavior,
- a way to keep greedy verification on the local-argmax path for speculative decode,
- or moving more post-lm-head/token-selection logic into one fused GPU-side operation so the framework does less per-token synchronization.
