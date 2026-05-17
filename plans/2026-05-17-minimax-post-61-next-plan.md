# MiniMax Post-61 tok/s Next Plan

Date: 2026-05-17

## Current Frame

The current promoted MiniMax M2.7 AutoRound result is no longer a theoretical
target: it is an honest, runtime-guarded baseline.

- Promoted baseline: `61.317497` output tok/s, `81.756663` total tok/s
- LocalMaxxing: `cmp9q9fzn04cto401tjcila06`
- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU, TP4, llm-scaler INT4 MoE path
- Quality: raw145 n64/n256 exact token hashes, semantic suite, and arithmetic
  repeat suite all pass

The working path is greedy local argmax over vocab-parallel logits. It avoids
full-vocab logits gather, preserves target greedy tokens, and uses the proven
pair all-gather reducer.

## Recent Findings

- Packed `int64` allreduce argmax is rejected. A standalone XCCL probe shows
  `ReduceOp.MAX` does not return correct packed maxima on this B70 stack.
- Direct `dist.all_gather_into_tensor` local argmax is quality-valid but slower
  than the promoted pair all-gather route: `61.086391` output tok/s versus
  `61.317497`.
- Gather/broadcast remains rejected because the actual runtime path hangs or
  fails under guard; the earlier row using that label was corrected by erratum.

## Quality Rules

Every speed candidate must pass before benchmarking:

- runtime import guard against the active installed vLLM files
- required marker for the exact code path being tested
- raw145 n64 exact combined token hash
- raw145 n256 exact combined token hash
- semantic suite with PASS, arithmetic `42`, and `add_one`
- repeated arithmetic greedy run to catch graph replay or request-state drift

LocalMaxxing submission requires both quality pass and a result worth sharing:

- material improvement over the current promoted baseline, or
- a separately labeled diagnostic that is clearly useful to others

No lower-quality quantization, expert dropping, unverified speculation, or
skipped Q/K RMS variance allreduce counts as an improvement.

## Next Workstreams

1. Build a standalone-probed custom pair reducer

   The current speed ceiling at the logits stage is communication/framework
   overhead after each LM-head local max. The custom path should compare
   `(float32 value, int32 token)` pairs directly and return the global greedy
   token without relying on XCCL `ReduceOp.MAX`.

   First probe outside vLLM:

   - four XPU ranks
   - deterministic cases with rank winners, negative values, ties, and random
     logits
   - exact agreement with an all-gather reference
   - stable repeated runs in one process launch

   Only after that, wire it into `LogitsProcessor.get_top_tokens` behind a new
   environment flag and runtime marker.

2. Keep the pair all-gather route as the fallback

   Do not remove the current promoted branch. New candidates should be easy to
   disable, and the strict wrapper should keep requiring markers so accidental
   source/runtime drift cannot create false wins.

3. Measure decode overhead without perturbing the benchmark

   The next timing pass should avoid synchronized full-model hooks. Useful
   low-risk measurements:

   - elapsed benchmark variance across repeated warm runs
   - runtime marker checks and AOT cache identity
   - small standalone collectives and reducer probes
   - log-only graph census around logits and final sampling

4. Revisit prefill only as a non-regressing side objective

   Prefill and total tok/s matter for sharing results, but decode tok/s remains
   the primary single-session target. Prefill changes should be kept if they
   improve total tok/s without reducing decode throughput or quality.

5. Speculation remains a separate verified track

   N-gram, suffix, and DFlash attempts have not yet produced a general
   quality-preserving win on this XPU stack. Future speculation work must report
   target verification and acceptance behavior, not only output speed.

## Immediate Actions

1. Publish the direct-gather no-improvement note and data.
2. Add a standalone reducer correctness probe for the next pair-reduction
   strategy before touching vLLM runtime behavior.
3. If the probe passes, add an env-gated vLLM candidate path with a distinct
   marker.
4. Run strict quality gates.
5. Benchmark at p512/n1536 with at least two repeats only after quality passes.
6. Submit to LocalMaxxing only if the result beats the promoted `61.317497`
   output tok/s baseline or is otherwise clearly useful as a labeled diagnostic.
