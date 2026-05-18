# MiniMax Decode Boundary Timing

Date: 2026-05-18

## Purpose

After the logits-to-work-sharing strict win, the next question was where to
spend optimization effort without changing quality. I added temporary timing
labels around MiniMax residual allreduce, MoE output allreduce, final norm, and
final logits boundaries, then reverted the model-forward labels after proving
they perturb the compiled graph.

The active runtime is back to the promoted hashes:

- `minimax_m2.py`: `e1408e4c95a662f43aca5712d30298bf75073626ffb10fcdce621c175888779f`
- `logits_processor.py`: `ccfe30c1469630aed3040de63970470db6110325850a552a0287764032bbc733`

## Findings

Compiled graph timing with synchronized post-forward labels, p512/n384:

- `minimax.final_logits`: `0.866055 ms/token`
- `logits.local_argmax_lm_head`: `0.600119 ms/token`
- `logits.gather_logits`: `0.135818 ms/token`
- `gpu_model_runner.async_output_tolist`: `0.041435 ms/token`

This means the full-logits path is a real but bounded post-forward cost. The
local lm-head projection is larger than TP logits gathering.

Eager timing with synchronized per-layer labels, p512/n48 after adding a MoE
allreduce label:

- `all_reduce:minimax_qk_var:(1, 2):torch.float32`: `0.096927 ms`
- `all_reduce:minimax.attn.delayed_residual_allreduce:(1, 3072):torch.float16`: `0.095092 ms`
- `all_reduce:minimax.moe.experts_allreduce:(1, 3072):torch.float16`: `0.092581 ms`
- `minimax.final_logits`: `0.842722 ms/token`
- `logits.local_argmax_lm_head`: `0.585511 ms/token`
- `logits.gather_logits`: `0.134978 ms/token`

The per-layer steady decode communication has three similar recurring costs:
Q/K variance allreduce, attention delayed residual allreduce, and MoE output
allreduce. The MoE label confirmed that the previously unlabeled hidden-size
collective was the MoE expert output reduction.

## Instrumentation Safety

A first attempt kept `allreduce_label(...)` wrappers in the compiled MiniMax
model path even when timing was disabled. That was not neutral:

- p512/n1536 no-timing speed check fell to `63.559813` output tok/s.
- A gated variant was still slow at `62.139637` output tok/s.

After reverting the MiniMax model file and restoring the logits processor hash,
the same promoted cache root produced:

- p512/n1536: `82.454765` output tok/s, `109.939687` total tok/s

Conclusion: timing wrappers inside the compiled model can alter the generated
graph even when their runtime behavior is intended to be inert. Keep future
diagnostics in temporary patches only, and verify the active file hashes before
any promoted benchmark.

## Decision

Do not promote or submit the diagnostic runs to LocalMaxxing. They are not speed
benchmarks because synchronized timing and eager mode intentionally distort
throughput.

Next optimization target should be a narrow quality-preserving branch around one
of these boundaries:

- reduce or fuse the full-logits lm-head/postprocess path without changing token
  selection;
- reduce the number of hidden-size collectives only if exact output quality can
  be preserved;
- improve the MoE expert output epilogue or scheduling, because MoE remains the
  largest eager per-layer region even after the logits-to-work-sharing win.

Speculative decode remains separate and must pass the same strict quality gates
before it can be considered a promoted path.
