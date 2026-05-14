# MiniMax AutoRound Quality Blocker

Date: 2026-05-13

While revalidating the post-reset MiniMax M2.7 AutoRound TP4 path, a speed-only
direct-load repeat produced a candidate `73.64` output tok/s result. The
promotion gate rejected it after the generation smoke exposed a more important
problem: the vLLM AutoRound path generated token id `0` repeatedly instead of
valid text.

## What Changed

`scripts/run-vllm-minimax-quality-check.py` now rejects degenerate generated
output. The old gate only checked determinism, so all-zero outputs could pass
as deterministic. The updated check records and fails on:

- only one distinct generated token id;
- zero printable non-space characters in decoded text.

The script also now exposes:

- `--temperature`, `--top-p`, `--top-k`;
- `--disable-custom-all-reduce`;
- `--disable-llm-scaler-moe`;
- `--allow-degenerate-output` for explicit diagnostics only.

## Probes

Prompt:

```text
Write one short sentence about PCIe tensor parallel bottlenecks.
```

| Probe | Graph | llm-scaler MoE | Output | Decision |
| --- | --- | --- | --- | --- |
| benchmark-mirrored path | yes | yes | all token id `0` | fail |
| eager control | no | no | all token id `0` | fail |
| sampled graph probe, `temp=1.0 top_p=0.95 top_k=40` | yes | yes | hung after AOT load | no result |

Artifacts:

- graph benchmark-mirrored:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/raw-quality-smoke-benchmirror-20260514T0121.json`
- eager no-llm-scaler control:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/raw-quality-smoke-eager-nollmscaler-20260514T0124.json`

Both completed probes had:

- `distinct_generated_token_count: 1`
- `first_distinct_generated_tokens: [0]`
- `printable_nonspace_text_chars: 0`
- `degenerate_output: true`

Token id `0` decodes to the NUL character for this tokenizer.

## Interpretation

The current vLLM `0.20.1-local` + MiniMax AutoRound W4A16 path is not
quality-cleared. This is not just the llm-scaler INT4 MoE fast path: disabling
llm-scaler and running without XPU graph still produced the same token-0
output.

Until this is fixed, MiniMax AutoRound throughput numbers should be treated as
speed-only engine diagnostics, not valid model-quality benchmark results. Do
not submit new MiniMax AutoRound vLLM speed highs to LocalMaxxing as promoted
results unless the strengthened quality smoke passes.

## Next Work

- Compare against a known-good MiniMax path:
  - GGUF UD-IQ4_XS through llama.cpp/SYCL if it can generate valid text;
  - SGLang with the AutoRound checkpoint if install/runtime support is viable;
  - an official FP8/BF16 MiniMax path only if memory permits.
- Inspect the vLLM MiniMax + INC AutoRound logits path:
  - confirm whether logits before sampling are finite and non-degenerate;
  - check final norm / `lm_head` execution and tensor-parallel gather;
  - verify whether AutoRound W4A16 MiniMax support is tested on XPU or only on
    CUDA in the upstream examples.
- Keep random-token throughput runs useful only as low-level scheduling and
  communication diagnostics until a text-generation correctness path passes.

