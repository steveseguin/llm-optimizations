# MiniMax M2.7 XPU Compiler-Pass Screens

Date: 2026-05-18

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM `0.20.1-local`, XPU TP4

Baseline for comparison: current strict promoted logits-to-work-sharing recipe, `82.404268` output tok/s and `109.872357` total tok/s mean at p512/n1536, ctx2048, MBT512, block256.

## Summary

The general vLLM compiler/communication fusion passes are not directly usable on this XPU stack yet:

- `fuse_allreduce_rms` fails before quality generation because the pass path asserts CUDA availability.
- `fuse_gemm_comms` is disabled by default for MiniMax hidden size unless `sp_min_token_num` is forced.
- Forced `fuse_gemm_comms` activates, but fails during AOT compile because `AsyncTPPass` is referenced on XPU without being imported.
- Forced sequence parallelism alone activates, but removes batch sizes `[1, 2]`, leaves graph capture sizes empty, and fails vLLM graph sizing.
- MiniMax-specific `fuse_minimax_qk_norm` with the XPU helper is quality-safe but slower than the current promoted recipe.

These are useful map points, not promoted benchmark results.

## MiniMax Q/K Norm Pass

Candidate:

```bash
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_minimax_qk_norm":true}}'
VLLM_MINIMAX_QK_NORM_XPU_HELPER_FUSION=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
```

Quality passed:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- 16-repeat arithmetic: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

Benchmark result:

- Repeats: `78.865249`, `79.378928` output tok/s
- Mean: `79.122088` output tok/s, `105.496118` total tok/s
- Delta vs promoted: about `-3.98%` output tok/s

Decision: reject for performance. The pass is useful as a quality-safe XPU helper proof, but it does not remove enough Q/K collective overhead and appears slower than the direct promoted implementation.

Artifacts:

- `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-qk-pass-helper-ar-20260518-strict-tp4-ctx2048-mbt512-bs256-20260518T191730Z-summary.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T193306Z.json`
- `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260518T193558Z.json`

## fuse_allreduce_rms

Candidate:

```bash
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}'
VLLM_XPU_EXPERIMENTAL_FUSE_ALLREDUCE_RMS=1
```

Outcome: failed before raw145 n64 output.

Key error:

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
Assertion failed, Torch not compiled with CUDA enabled
```

Decision: do not retest unchanged. This pass still reaches CUDA-only assumptions on the current XPU stack.

## fuse_gemm_comms

Unforced candidate:

```bash
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_gemm_comms":true}}'
VLLM_XPU_EXPERIMENTAL_FUSE_GEMM_COMMS=1
```

Outcome: vLLM disabled the pass:

```text
Model hidden_size too small for the SP threshold heuristic, disabling. To force SP, set pass_config.sp_min_token_num manually.
pass_config: enable_sp=false, fuse_gemm_comms=false
```

Forced candidate:

```bash
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"fuse_gemm_comms":true,"sp_min_token_num":1}}'
VLLM_XPU_EXPERIMENTAL_FUSE_GEMM_COMMS=1
```

Outcome: pass activated, then failed during compile/profile:

```text
Enabled custom fusions: gemm_comms
pass_config: enable_sp=true, fuse_gemm_comms=true, sp_min_token_num=1
NameError: name 'AsyncTPPass' is not defined
```

Likely root cause: `vllm/compilation/passes/pass_manager.py` references `AsyncTPPass` when `fuse_gemm_comms` is true, but the import is CUDA-gated in this build. Patching the import alone may still not be enough because the underlying collective fusion path is likely CUDA/symmetric-memory oriented.

## Sequence Parallelism Alone

Candidate:

```bash
COMPILATION_CONFIG_JSON='{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE","pass_config":{"enable_sp":true,"sp_min_token_num":1}}'
VLLM_XPU_EXPERIMENTAL_ENABLE_SP=1
```

Outcome: sequence parallelism activated, but batch-size filtering and graph sizing conflict for batch-1 decode:

```text
Batch sizes [1, 2] are removed because they are not multiple of tp_size 4 when sequence parallelism is enabled
cudagraph_capture_sizes=[]
max_cudagraph_capture_size=0
AssertionError: Maximum cudagraph size should be greater than or equal to 1 when using cuda graph.
```

Decision: do not retest unchanged. If this path is revisited, the next useful experiment is a deliberate SP configuration for batch sizes that are multiples of TP4, not the current single-session decode target.

## Next Direction

The compiler-pass screen says the easy vLLM communication-fusion toggles are not the path to the next MiniMax single-session win on B70. The next useful work should stay closer to XPU-native decode boundaries:

- keep the promoted exact router-logits WS path;
- instrument or prototype MoE/projection epilogue fusion where the work is already XPU-native;
- avoid graph scratch reuse unless buffers have graph-safe lifetime ownership;
- treat vLLM SP/gemm-comms as a future multi-sequence throughput path unless XPU pass support is repaired.
