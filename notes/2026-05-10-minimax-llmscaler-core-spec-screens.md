# MiniMax M2.7 AutoRound llm-scaler Core and Speculation Screens, 2026-05-10

## Current Orders

Primary objective:

- Optimize MiniMax M2.7 AutoRound INT4 on 4x Intel Arc Pro B70.
- Raise the aspiration now that we moved from GGUF capacity testing to the faster AutoRound path:
  - repeatable non-speculative target: `60+` output tok/s at p512/n1536
  - speculative target: `75+` output tok/s if target-model verification is preserved
- Preserve quality: no expert dropping, no skipped Q/K RMS variance allreduce, no non-equivalent root residual tricks, and no speculative result counted unless the target verifies the emitted tokens.
- Keep GPU power limits unchanged; focus on software/runtime/compiler/kernel work.
- Record benchmark results with both total/prefill-inclusive tok/s and output/decode tok/s.
- Submit only useful, valid, reproducible results to LocalMaxxing.
- Keep notes, data, patches, and reproduction details pushed to the private GitHub notes repo.

Secondary objectives:

- Use Qwen3.6/Qwen3.5 27B or 35B only as comparison probes when they isolate transferable dense-model TP, attention, KV, or graph-scheduling behavior.
- Continue to mine vLLM, Intel llm-scaler, SGLang, KTransformers, and related projects for design ideas, but let local B70 measurements decide what stays.

## Where We Are

Current clean quality-preserving MiniMax AutoRound reference:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM `0.20.1` local XPU runtime
- hardware: 4x B70 32GB
- recipe: TP4, FP16 dtype, llm-scaler INT4 MoE bridge, CCL P2P enabled, XPU graph disabled, no speculation
- p512/n1536 accepted long-run anchor: `37.552538` output tok/s, `50.070051` total tok/s
- current clean p512/n1536 refresh: `37.168339` output tok/s, `49.557786` total tok/s
- current p512/n512 clean refresh: `35.647520` output tok/s, `71.295039` total tok/s

The accepted anchor is a solid public reference, but it is still far below the revised `60+` output tok/s target. The remaining work is now lower-level: communication boundaries, Q/K RMS variance allreduce, attention/KV scheduling, and MoE/projection epilogues.

## llm-scaler Core ESIMD Build

I built the llm-scaler core ESIMD extension locally after fixing the compiler invocation to use the oneAPI compiler and the `-device bmg` form:

- source root: `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm`
- generated core extension: `custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so`
- existing MiniMax-useful extension kept in place: `moe_int4_ops.cpython-312-x86_64-linux-gnu.so`

Result: unsafe on this stack.

Observed failures:

- Direct calls to `esimd_fused_add_rms_norm` segfaulted in `libsycl.so.8`.
- A vLLM MiniMax smoke run with the generated core `.so` importable failed during worker import/SYCL registration, with stack frames in `sycl::_V1::detail::ProgramManager::addImage` and `__sycl_register_lib`.

Core failure log:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-smoke-after-esimd-core/vllm-minimax-m27-autoround-tp4-p64n32-20260510T184949Z.log`

Action taken:

- Quarantined only the generated core `.so` as:
  - `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/custom_esimd_kernels.cpython-312-x86_64-linux-gnu.so.disabled-20260510T185258Z`
- Left the working `moe_int4_ops` extension active.

Health check after quarantine:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-smoke-after-esimd-quarantine/vllm-minimax-m27-autoround-tp4-p64n32-20260510T185306Z.log`
- JSON: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-smoke-after-esimd-quarantine/vllm-minimax-m27-autoround-tp4-p64n32-20260510T185306Z.json`
- p64/n32 total: `77.330536` tok/s
- p64/n32 output-equivalent: `25.776845` tok/s

Decision:

- Keep llm-scaler core ESIMD fused add/RMS/GEMV disabled until we can fix the SYCL registration/call crash.
- Keep using the llm-scaler INT4 MoE path, which remains healthy.
- Do not use llm-scaler `resadd_norm_gemv_int4` for the MiniMax router. The MiniMax AutoRound config keeps the router/gate in 16-bit/float form, so quantizing that boundary is not a quality-preserving speedup without separate validation.

## Speculation Screens

Shape: TP4, p64/n128, `MAX_MODEL_LEN=512`, `MAX_BATCHED_TOKENS=256` unless noted, random benchmark payload.

| Run | Extra args | Total tok/s | Output tok/s | Outcome |
| --- | --- | ---: | ---: | --- |
| baseline | none | `50.618235` | `33.745490` | clean reference for this short screen |
| ngram | `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":1}` | `11.058862` | `7.372575` | negative; async scheduling disabled |
| ngram, larger token budget | same ngram config, `MAX_BATCHED_TOKENS=1024` | `11.129270` | `7.419513` | negative; scheduler budget was not the blocker |
| ngram_gpu | `{"method":"ngram_gpu","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":1}` | none | none | loaded/compiled, then stalled at 0 processed prompts until terminated |

Logs:

- baseline: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-spec-screen-baseline-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T185539Z.log`
- ngram: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-spec-screen-ngram-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T185814Z.log`
- ngram with larger token budget: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-spec-screen-ngram-p64n128-mbt1024/vllm-minimax-m27-autoround-tp4-p64n128-20260510T190154Z.log`
- ngram_gpu: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-spec-screen-ngramgpu-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T190545Z.log`

Decision:

- Plain ngram is closed for this random MiniMax workload on the current XPU stack.
- `ngram_gpu` is not usable yet; it avoids the immediate async-disable behavior but stalls before completing the benchmark.
- Keep speculation on the roadmap only with a target-compatible draft path, acceptance-rate instrumentation, and a completed benchmark. The stretch target remains `75+` output tok/s only if target verification is intact.

## Stock vLLM AllReduce+RMS Pass Screen

Command variant:

```bash
EXTRA_ARGS='-cc.pass_config.fuse_allreduce_rms=True'
INPUT_LEN=64 OUTPUT_LEN=32 MAX_MODEL_LEN=256 MAX_BATCHED_TOKENS=128 \
  /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Result:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-fuse-allreduce-rms-screen-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T191511Z.log`
- p64/n32 total: `80.285435` tok/s
- p64/n32 output-equivalent: `26.761812` tok/s

Important log line:

```text
Feature 'AllReduce + RMSNorm fusion' is not yet supported on XPU and will be disabled.
```

Decision:

- This is not a real XPU fusion result. The flag is accepted, then disabled by the XPU platform path.
- The next useful work is not flag tuning; it is implementing an XPU equivalent of the allreduce/residual/RMS or Q/K variance/RMS boundary.

## Enforce-Eager Screen

The Intel/vLLM AutoRound docs still say to add `--enforce-eager` for WnA16 Intel GPU/CPU deployment. I tested whether that is a compatibility-only recommendation or a throughput path for this MiniMax setup.

Command variant:

```bash
EXTRA_ARGS='--enforce-eager'
INPUT_LEN=64 OUTPUT_LEN=128 MAX_MODEL_LEN=512 MAX_BATCHED_TOKENS=256 \
  /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Result:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-enforce-eager-screen-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T191742Z.log`
- p64/n128 total: `27.010633` tok/s
- p64/n128 output-equivalent: `18.007089` tok/s
- baseline for same shape: `50.618235` total, `33.745490` output-equivalent

Decision:

- Keep `--enforce-eager` off for throughput benchmarks on this MiniMax B70 path. It may remain a compatibility flag for other Intel AutoRound recipes, but it is a clear slowdown here.

## External Notes

- vLLM's current fusion docs list `fuse_allreduce_rms` and MiniMax-specific `fuse_minimax_qk_norm`, but the support matrix is CUDA/ROCm-oriented and the local XPU runtime explicitly disables AllReduce+RMS. Source: https://docs.vllm.ai/en/stable/design/fusions/
- vLLM documents `ngram` speculation and says speculative decoding aims to preserve target behavior, but also warns that it is not generally optimized for all datasets. Source: https://docs.vllm.ai/en/v0.11.0/features/spec_decode.html
- vLLM's `ngram_gpu` proposer prepares speculative inputs on-device without CPU-GPU sync in its documented implementation, which makes it an interesting future path if the current XPU stall can be fixed. Source: https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/ngram_proposer_gpu/
- Intel llm-scaler documents OneCCL P2P/USM modes plus INT4/FP8 serving ideas. Locally, small-batch MiniMax decode still looks dominated more by graph/collective boundary placement than by raw oneCCL microbench latency. Source: https://github.com/intel/llm-scaler/blob/main/vllm/README.md

## Next Implementation Direction

The flag-level work is mostly exhausted. The next pass should focus on native XPU changes:

1. Prototype or port an XPU equivalent for the MiniMax Q/K variance allreduce plus RMSNorm boundary.
2. Prototype an XPU allreduce/residual/RMSNorm boundary at post-attention or post-MoE, preferably as C++/SYCL or an Inductor/backend pass rather than a Python custom op.
3. Keep the active runtime clean and env-gated; no experimental wrapper should affect baseline runs by default.
4. Build low-overhead diagnostics around attention/KV, Q/K RMS, projection, and MoE epilogues to choose the first fusion target.
5. Re-run p512/n512 and p512/n1536 after each source change, then submit to LocalMaxxing only when the result is repeatable and quality-preserving.
