# MiniMax M2.7 Allreduce Boundary Probes

Goal: reduce framework-visible TP collective boundaries on the 4x B70 MiniMax M2.7 AutoRound path without changing model weights, quantization, routing, sampler, KV dtype, or power limits.

Current accepted baseline remains the strict quality-valid no-async TP4 path at 62.662 output tok/s and 83.550 total tok/s (`cmp7sf8xo00dko4018nmrmckc` on LocalMaxxing).

## What Was Tried

### Custom Allreduce + Sequence Parallelism

Command shape:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1 \
VLLM_XPU_EXPERIMENTAL_ENABLE_SP=1 \
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"enable_sp":true,"sp_min_token_num":1}}'
```

Result: rejected before tokens. The custom-op pattern avoided the previous torchbind pattern-construction crash, but the stock sequence-parallel rewrite failed in Inductor:

```text
RuntimeError: The size of tensor a (s72) must match the size of tensor b ((s72//4)) at non-singleton dimension 0
```

Interpretation: stock sequence parallelism is not residual-shape-safe for this MiniMax decode graph. It reduces/scatters the token dimension while later residual users still expect the full token dimension.

### Custom Allreduce Only

Command shape:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
```

Result: rejected at the first exact token-hash gate. Output was deterministic and non-degenerate, but the raw145 n64 hash changed from expected `267cbf...5bd` to `0ada9c...e20`.

PyTorch also warned that `vllm::all_reduce` may alias its input despite the custom op schema expecting non-aliasing output. That makes this path unsafe until the operator mutation/alias contract is fixed.

### Functional Out-Of-Place Allreduce

Command shape:

```bash
VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1
```

Result: rejected at semantic repeat gate. It passed both exact raw prompt hashes:

- raw145 n64: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`

But the semantic suite failed because two greedy repeats produced nondeterministic token hashes, even though the required substrings and regexes were present.

## Decision

Do not submit any of these to LocalMaxxing. None passed the full quality gate.

Keep the accepted no-clone c10d path as the promoted baseline. The useful learning is that:

- sequence parallelism needs a MiniMax-specific residual-shape-safe rewrite;
- custom allreduce needs correct mutation/alias semantics before quality testing matters;
- functional out-of-place allreduce is closer, but still not deterministic enough.

Next productive direction remains true fused kernels around the existing accepted math: Q/K RMS variance allreduce fusion, hidden-state allreduce plus RMS/add fusion, or MoE epilogue fusion.
