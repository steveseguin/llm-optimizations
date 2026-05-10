# MiniMax M2.7 AutoRound DBO, DCP, and EP Round-Robin Screens, 2026-05-10

## Context

Target remains quality-preserving MiniMax M2.7 AutoRound INT4 on 4x B70:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- runtime: active vLLM/XPU venv, `v0.20.1`
- path: TP4, llm-scaler MoE bridge enabled, FP16 runtime dtype
- power limits: unchanged
- quality guardrail: no expert dropping, no skipped Q/K variance allreduce, no speculative shortcut without target verification

The revised aspiration target is `60+` output tok/s at p512/n1536. These screens tested flag-level paths before returning to lower-level collective and fusion work.

## Web/Upstream Notes

- vLLM DBO documentation says DBO must be used with `--data-parallel-size > 1`, `--enable-expert-parallel`, and DeepEP all-to-all. The current B70/XPU stack does not have DeepEP, so DBO is useful as a design reference for overlap but not as a direct flag-level solution here.
  - source: https://docs.vllm.ai/en/v0.17.1/design/dbo/
- vLLM speculative decoding docs list `draft_model`, `ngram`, `suffix`, `mtp`, `eagle3`, and `dflash`, and keep the result target-verified. That keeps speculation on the roadmap, but earlier MiniMax n-gram/ngram_gpu/DFlash runs remain negative or blocked for this harness.
  - source: https://docs.vllm.ai/en/v0.20.1/features/speculative_decoding/
- Intel llm-scaler documents OneCCL P2P/USM switching and notes that small batches show minimal difference, while larger batches may benefit more from P2P. That matches our local result: oneCCL microbench latency is not the only decode limiter; graph and boundary placement matter.
  - source: https://github.com/intel/llm-scaler/blob/main/vllm/README.md

## DBO Screen

Command variant:

```bash
EXTRA_ARGS='--enable-dbo --dbo-prefill-token-threshold 1 --dbo-decode-token-threshold 1'
INPUT_LEN=64 OUTPUT_LEN=32 MAX_MODEL_LEN=128 MAX_BATCHED_TOKENS=128 \
  /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Outcome: blocked before model execution.

Log:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dbo-screen/vllm-minimax-m27-autoround-tp4-p64n32-20260510T182904Z.log`

Failure:

```text
Microbatching currently only supports the deepep_low_latency and deepep_high_throughput all2all backend.
```

Decision: do not spend more time on vLLM DBO flags on B70 until there is an XPU all-to-all backend with the same overlap hooks as DeepEP, or until we implement a B70-specific overlap path.

## Decode Context Parallel Screen

Command variant:

```bash
EXTRA_ARGS='--decode-context-parallel-size 2'
INPUT_LEN=64 OUTPUT_LEN=32 MAX_MODEL_LEN=128 MAX_BATCHED_TOKENS=128 \
  /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Outcome: blocked during config validation.

Log:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dcp-screen/vllm-minimax-m27-autoround-tp4-p64n32-20260510T183533Z.log`

Failure:

```text
tensor parallel size 4 must be greater than total num kv heads 8 when enable decode context parallel for GQA/MQA
```

Decision: DCP is not a viable TP4 MiniMax flag path on this four-B70 host.

## Expert Parallel Round-Robin Screen

Command variant:

```bash
EXTRA_ARGS='--enable-expert-parallel --expert-placement-strategy round_robin'
INPUT_LEN=64 OUTPUT_LEN=32 MAX_MODEL_LEN=128 MAX_BATCHED_TOKENS=128 \
  /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Cold run:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ep-roundrobin/vllm-minimax-m27-autoround-tp4-p64n32-20260510T183006Z.log`
- JSON: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ep-roundrobin/vllm-minimax-m27-autoround-tp4-p64n32-20260510T183006Z.json`
- total: `20.839690709851368` tok/s
- output-equivalent: `6.946563569950456` tok/s

Warm run, same AOT cache:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ep-roundrobin/vllm-minimax-m27-autoround-tp4-p64n32-20260510T183303Z.log`
- JSON: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ep-roundrobin/vllm-minimax-m27-autoround-tp4-p64n32-20260510T183303Z.json`
- total: `67.01469435826483` tok/s
- output-equivalent: `22.338231452754943` tok/s

Decision: negative. This is consistent with the earlier quality-preserving EP4 p512/n1536 diagnostic, which was slower than the TP4 reference. Round-robin placement is not a route to the 60 tok/s target for batch-1 MiniMax on this host.

## Updated AOT Census Tooling

`scripts/summarize-vllm-aot-collectives.sh` now counts:

- `_c10d_functional` allreduce comments and calls
- `vllm.all_reduce` custom-op calls
- allreduce placeholders in compiled subgraphs
- local allreduce-to-RMS/MoE fused boundary kernels
- INT4 RMS/GEMM fused kernels

Current clean TP4 p512/n1536 AOT hash:

- `/home/steve/.cache/vllm/torch_compile_cache/torch_aot_compile/679011672fb322d8dc186c528582d5f2bee43d3132510c02990580dbc9a4ccbf/inductor_cache`
- `all_reduce_comment_lines=92`
- `allreduce_rms_moe_boundary_lines=48`
- `compiled_int4_rms_kernels=20`
- `compiled_ar_rms_moe_kernels=28`

The census shows Inductor is already fusing useful local residual/RMS/MoE pointwise work after the TP collective. The remaining target is not another Python wrapper around allreduce; it is a real C++/SYCL or backend-level collective boundary that removes the collective scheduling/fencing cost or overlaps it with adjacent work.

## Next Step

Return to lower-level implementation:

1. Keep DBO/DCP/EP-round-robin closed as flag-level paths.
2. Use the AOT census to target the repeated allreduce-to-RMS/MoE boundary.
3. Prototype only C++/SYCL or compiler/backend-level changes for this boundary; previous Python and opaque custom-op wrappers are negative.
4. Re-run p512/n512 and p512/n1536 after any source change and submit only repeatable quality-preserving improvements.
