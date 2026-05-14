# MiniMax AutoRound Compile Correctness Split

Date: 2026-05-13

## Summary

The MiniMax M2.7 AutoRound W4A16 path is not intrinsically broken on the
4x B70 box. The failure is specifically tied to vLLM compiled execution.

Control results:

| Path | Result | Decision |
| --- | --- | --- |
| GGUF UD-IQ4_XS RPC+SYCL, greedy smoke | Valid printable text, `17.29` decode tok/s on 16-token smoke | Hardware and MiniMax prompt path sane |
| AutoRound vLLM, compiled, no llm-scaler MoE, custom allreduce disabled | 32 copies of token id `0` | Fail |
| AutoRound vLLM, `enforce_eager=True`, no llm-scaler MoE, custom allreduce disabled | Valid printable text | Pass |
| AutoRound vLLM, `enforce_eager=True`, llm-scaler MoE enabled, custom allreduce enabled | Same valid tokens/text as disabled-control | Pass |
| AutoRound vLLM, compiled, `use_inductor_graph_partition` disabled | 32 copies of token id `0` | Fail |
| AutoRound vLLM, compiled, forced `ir_op_priority.rms_norm=xpu_kernels,native` | 32 copies of token id `0` | Fail |
| AutoRound vLLM, `STOCK_TORCH_COMPILE` | Hung in shared-memory broadcast/compile phase after load; killed | No result |

## Key Artifacts

- GGUF control:
  `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/correctness/gguf-rpc-y2-greedy-smoke-20260514T014907Z`
- Compiled failure, no llm-scaler/custom allreduce:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/eager-nollmscaler-quality-clean-20260514T015550Z.json`
- Eager pass, no llm-scaler/custom allreduce:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/eager-enforced-nollmscaler-quality-20260514T020126Z.json`
- Eager pass, optimized llm-scaler/custom allreduce:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/eager-enforced-optimized-quality-20260514T020342Z.json`
- Compiled failure, graph partition disabled:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-nopartition-quality-20260514T020619Z.json`
- Compiled failure, XPU RMS priority forced:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-xpurms-quality-20260514T021054Z.json`
- Stock compile hang log:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/stockcompile-quality-20260514T021816Z.log`

## Interpretation

The previous all-zero AutoRound quality failures were not caused by stale vLLM
workers, llm-scaler MoE, custom allreduce, prompt rendering, tokenizer decode,
or the AutoRound weights themselves. Clean `enforce_eager=True` TP4 generation
produced the same sensible text with and without llm-scaler/custom allreduce:

```text
Paris.

The capital of Germany is Berlin.

The capital of Italy is Rome.

The capital of Spain is Madrid.

The capital of Portugal is Lisbon.
```

The compiled path fails even when:

- XPU graph is disabled;
- llm-scaler MoE is disabled;
- custom allreduce is disabled;
- `use_inductor_graph_partition` is disabled;
- RMSNorm IR priority is forced to `xpu_kernels,native`.

That leaves a broader vLLM/Inductor compiled-forward correctness issue for this
MiniMax AutoRound/XPU shape. The symptom is deterministic collapse to token id
`0`, which decodes to NUL for this tokenizer.

## Decision

Do not promote or submit compiled AutoRound MiniMax throughput results to
LocalMaxxing until the strengthened quality gate passes. The quality-cleared
AutoRound path today is `enforce_eager=True`, but its short smoke decode speed
was only about `10.5-13.1` tok/s and is not competitive with the GGUF baseline.

## Next Work

1. Add a logits/debug hook around vLLM compiled versus eager execution and
   compare pre-sampler logits for the same one-token prompt.
2. Bisect compiled-forward regions by disabling or excluding candidate
   components from Inductor: final norm/lm_head, attention output, Q/K RMS,
   and MiniMax MoE blocks.
3. Inspect the generated AOT graph for the compiled failure and look for missing
   final logits/gather, wrong output aliasing, or an all-zero/NaN-masked logits
   buffer before sampling.
4. Keep `enforce_eager=True` as the correctness oracle for AutoRound while
   continuing GGUF RPC+SYCL as an independent MiniMax control.
