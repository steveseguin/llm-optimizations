# MiniMax oneCCL Copy-Engine Screens, 2026-05-13

## Purpose

Screen a quality-neutral oneCCL runtime path against the current MiniMax M2.7
AutoRound best recipe. Intel documents that GPU-buffer `topo` collectives can
use either compute kernels or copy engines for the scale-up phases:

- `CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=0` uses copy engines for the
  allreduce reduce-scatter phase.
- `CCL_ALLGATHERV_MONOLITHIC_PIPELINE_KERNEL=0` uses copy engines for the
  allreduce allgather phase.

Source:
https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-15/environment-variables.html

Quality guardrail: model weights, quantization, dtype, KV dtype, sampler,
routing, tensor parallelism, graph recipe, and power settings were unchanged.

## Baseline

Current promoted recipe:

- output tok/s: `73.306312`
- total tok/s: `97.741749`
- AOT hash:
  `d0fed86b5a7cf64dcdb3e82d0b24effed00520ebd7f6018c6244d82201e6a98c`
- recipe: TP4 FP16, llm-scaler INT4 MoE, XPU graph, MiniMax attention delayed
  allreduce, `--block-size 256`, `MAX_BATCHED_TOKENS=512`, prefix caching off

## Results

| Screen | Output tok/s | Total tok/s | Decision |
| --- | ---: | ---: | --- |
| both copy-engine phases | `73.485844` | `97.981125` | not repeatable |
| both copy-engine phases repeat 1 | `72.973966` | `97.298621` | negative |
| both copy-engine phases repeat 2 | `72.625227` | `96.833637` | negative |
| reduce-scatter copy-engine only | `71.757088` | `95.676117` | negative |
| allgather copy-engine only | `72.793538` | `97.058051` | negative |
| `CCL_SYCL_OUTPUT_EVENT=0` | `72.942286` | `97.256381` | negative |

The first combined run was `+0.179532` output tok/s over the current best, but
the next two combined runs fell below it. The split-phase screens were also
below current best, especially reduce-scatter-only.

## Decision

Keep oneCCL defaults for the current MiniMax single-session path. The copy-engine
settings and `CCL_SYCL_OUTPUT_EVENT=0` are quality-neutral and worth knowing
about, but they do not provide a repeatable sustained decode win on this B70
stack.

No LocalMaxxing submission: the only above-best number was not repeatable.

## Artifacts

- Data summary:
  `data/minimax-m27-onccl-copy-engine-screens-20260513.json`
- Best single copy-engine log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T200259Z.log`
- Reduce-scatter-only log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T201717Z.log`
- Allgather-only log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T201959Z.log`
- `CCL_SYCL_OUTPUT_EVENT=0` log:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260513T202414Z.log`
