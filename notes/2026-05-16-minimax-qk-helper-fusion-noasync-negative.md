# MiniMax M2.7 Q/K Helper Fusion Retest: Rejected

Date: 2026-05-16

## Goal

Retest the MiniMax Q/K RMS fusion path under the stricter no-async recipe:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1
VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1
QUALITY_ASYNC_SCHEDULING=off
BENCH_ASYNC_SCHEDULING=off
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}'
RUN_EXTENDED_QUALITY=1
```

The hypothesis was that reducing framework callbacks around Q/K variance and
RMS application could improve decode speed without changing quality.

## Result

Rejected. The candidate passed the two exact raw canaries, but failed the
semantic suite because two greedy repeats were not token-deterministic.

- raw145 n64 exact token hash: pass
- raw145 n256 exact token hash: pass
- semantic PASS/42/add_one suite: fail, nondeterministic token hashes
- benchmark: not run, because quality gate failed

The semantic outputs still contained the required content, but one repeat
returned `"\n\n42"` and another returned `"\n\n\n42"` for the arithmetic canary.
That is a small textual difference, but it is enough to reject the path because
repeatability matters for publishable results.

## AOT Collective Census

The fusion pass also did not change the graph boundary shape:

```text
actual_allreduce_call_lines: 1496
actual_wait_tensor_call_lines: 1496
actual_allreduce_wait_pairs_within_7_lines: 1496
attention_o_proj_hidden: 496
embedding_hidden: 8
moe_hidden: 496
qk_rms_variance: 496
```

This matches the previous boundary census. In other words, the pass did not
remove the Q/K variance collectives or their immediate waits. It is not the
right lever for the current bottleneck.

## Artifacts

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-qk-helper-fusion-noasync-strict-tp4-ctx2048-mbt512-bs256-20260516T195029Z-summary.json`
- Captured data: `data/minimax-m27-qk-helper-fusion-noasync-negative-20260516.json`
- Strict-gate patch: `patches/minimax-strict-gate-compilation-config-json-20260516.patch`
- AOT cache: `/mnt/fast-ai/vllm-cache-exp/minimax-qk-helper-fusion-noasync-20260516T195029Z/torch_compile_cache/torch_aot_compile/dffa4213474dcaf63e2feca57a7dad4ad4cfab759a74c9f17bfe4beb48537de0`

## Next Direction

The useful target is not this Python-level Q/K helper pass. The next useful
work is an XPU-native fused collective path that actually removes boundaries:

- Q/K variance allreduce + apply RMS: replace `var -> allreduce/wait -> apply`
  with one graph-safe XPU custom op or a capture-safe pair that does not force
  an immediate framework wait.
- Hidden allreduce + RMSNorm / MoE consumer: build an XPU equivalent of the
  CUDA/FlashInfer `fuse_allreduce_rms` path rather than enabling the existing
  CUDA-oriented pass.
- Keep the no-async scheduling reliability rule until an extended sixpack
  canary proves otherwise.
