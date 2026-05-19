# MiniMax M2.7 Allreduce Shape Log and n2048 Negative

Date: 2026-05-19

## Summary

Added a guarded allreduce shape logger and used it to explain why the earlier generic `<=4096` in-place allreduce threshold slowed down MiniMax M2.7 AutoRound on 4x B70.

The first graph-mode diagnostic attempted to log from inside a `torch.compile` trace and failed with:

- `torch._dynamo.exc.Unsupported: logging.Logger method not supported for non-export cases`

The logger was then guarded with `torch.compiler.is_compiling()` / `torch._dynamo.is_compiling()`, and an eager diagnostic completed successfully.

## Shape Findings

Eager shape logging on the promoted recipe found these decode-relevant allreduce shapes:

- Profile/prefill hidden-state allreduce: dtype `torch.float16`, shape `(512, 3072)`, numel `1572864`
- Profile/prefill Q/K RMS variance allreduce: dtype `torch.float32`, shape `(512, 2)`, numel `1024`
- Decode hidden-state allreduce: dtype `torch.float16`, shape `(1, 3072)`, numel `3072`
- Decode Q/K RMS variance allreduce: dtype `torch.float32`, shape `(1, 2)`, numel `2`
- 42-token prompt hidden-state allreduce: dtype `torch.float16`, shape `(42, 3072)`, numel `129024`
- 42-token prompt Q/K RMS variance allreduce: dtype `torch.float32`, shape `(42, 2)`, numel `84`

This explains the `n4096` result: `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=4096` routes the fp16 decode hidden-state `(1, 3072)` allreduce through the mutating custom op. That path passed quality but produced repeated `ocloc`/Triton compile fallback noise and slower performance.

## n2048 Candidate

Tested a narrower threshold:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=2048`
- `VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=0`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_WS=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE_MINIMAX_LOGITS_WS=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=0`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
- `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `CCL_TOPO_P2P_ACCESS=1`

This should catch fp32 Q/K variance collectives up to the `(512, 2)` profile/prefill shape while avoiding the fp16 `(1, 3072)` decode hidden-state allreduce.

## Quality

The n2048 candidate passed the full strict gate:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- arithmetic repeat n64 r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64 r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4 on 4x Intel Arc Pro B70 32GB.

Four repeats after quality pass:

- Output tok/s: `87.143737`, `87.209267`, `87.367663`, `86.702200`
- Total tok/s: `116.191650`, `116.279023`, `116.490217`, `115.602933`
- Mean output tok/s: `87.105717`
- Mean total tok/s: `116.140956`

Comparison:

- Versus alias-correct tiny-FP32 in-place result (`88.103866` output tok/s): `-1.13%`
- Versus fastest tiny-FP32 skip-clone headline (`88.748424` output tok/s): `-1.85%`
- Versus generic n4096 threshold (`86.386284` output tok/s): `+0.83%`

## Decision

Reject and do not submit to LocalMaxxing. The result is quality-safe and cleaner than n4096, but still slower than the current alias-correct tiny-FP32 in-place baseline.

The useful lesson is that generic numel thresholding is too blunt. The best current path remains a dtype-specific tiny-FP32 in-place route for Q/K variance scalar decode, while broader gains require true boundary fusion or shape-specific communication handling rather than making all `<=2048` collectives in-place.

## Artifacts

- Shape diagnostic JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-allreduce-shape-log-eager-promoted-20260519.json`
- Shape diagnostic log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-allreduce-shape-log-eager-promoted-20260519.log`
- n2048 summary JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-allreduce-inplace-n2048-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T034737Z-summary.json`
- n2048 quality dir: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-allreduce-inplace-n2048-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T034737Z-quality`
- n2048 bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T040325Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T040617Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T040903Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T041157Z.json`
- Data file: `data/minimax-m27-allreduce-shape-log-and-n2048-negative-20260519.json`
- Patch note: `patches/minimax-allreduce-shape-logger-and-n2048-screen-20260519.patch`

## Next Step

Keep `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=0` for promoted runs and keep the alias-correct `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=2` path as the reproduction-safe baseline.

Next optimization work should target one of:

- A shape/dtype-specific Q/K variance allreduce fusion that never touches fp16 hidden-state collectives.
- A fused residual/add/norm or residual/allreduce boundary that removes one decode-time framework communication boundary.
- A communication benchmark for `(1, 2)`, `(1, 3072)`, `(512, 2)`, and `(512, 3072)` to separate CCL latency from graph/compiler overhead.
