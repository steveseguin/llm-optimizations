# MiniMax M2.7 Generic In-Place Allreduce Threshold Negative

Date: 2026-05-19

## Summary

Tested a broader decode-sized in-place custom allreduce route:

- `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
- `VLLM_XPU_CUSTOM_ALLREDUCE_SKIP_CLONE_FP32_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_TINY_FP32_INPLACE_MAX_NUMEL=0`
- `VLLM_XPU_CUSTOM_ALLREDUCE_INPLACE_MAX_NUMEL=4096`
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

This extends the alias-correct tiny-FP32 in-place custom op path by allowing any tensor with `numel <= 4096` to route through the mutating no-return allreduce op. The intent was to remove more graph-visible clone/out-of-place overhead while preserving PyTorch alias correctness.

## Quality

The candidate passed the strict quality gate before benchmarking:

- raw145 n64 exact: `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`
- raw145 n256 exact: `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537`
- semantic suite: `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805`
- arithmetic repeat n64 r16: `578ec378bf31cb16fb49ac5c0043270fd00a0f7898e18ac498a41ffe775d7994`
- extended sixpack n64 r2: `1e3560554f57b2b56cec8f49f28bc8ba12e9e0ced26bdc99a976f1433c99caa7`

## Performance

Shape: p512/n1536, ctx2048, batch 1, TP4 on 4x Intel Arc Pro B70 32GB.

Four manual repeats after quality pass:

- Output tok/s: `87.525933`, `86.836183`, `85.303217`, `85.879804`
- Total tok/s: `116.701244`, `115.781578`, `113.737623`, `114.506406`
- Mean output tok/s: `86.386284`
- Mean total tok/s: `115.181713`

Comparison:

- Versus alias-correct tiny-FP32 in-place result (`88.103866` output tok/s): `-1.95%`
- Versus fastest tiny-FP32 skip-clone headline (`88.748424` output tok/s): `-2.66%`

## Observations

This result is quality-safe but slower, so it is rejected and was not submitted to LocalMaxxing.

Logs repeatedly showed `Triton compilation failed: triton_red_fused__to_copy_mm_t_9`, followed by `ocloc` error code `245` and `Build failed with error code: -11`. The runs still produced valid JSON and passed quality, but the fallback noise is a strong signal that the broad `<=4096` route is pushing a decode-sized reduction into a worse compiler path.

The best inference is that the tiny FP32 scalar path is a good target for in-place mutation, while broader hidden-state or projection-sized collectives need shape-specific routing or true fusion. Do not promote the generic threshold.

## Artifacts

- Summary JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-allreduce-inplace-n4096-quality-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T025834Z-summary.json`
- Quality dir: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-allreduce-inplace-n4096-quality-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T025834Z-quality`
- Bench JSONs:
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T031500Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T031749Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T032612Z.json`
  - `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260519T032908Z.json`
- Data file: `data/minimax-m27-allreduce-inplace-n4096-negative-20260519.json`
- Patch note: `patches/minimax-allreduce-inplace-threshold-20260519.patch`

## Next Step

Do not continue broad thresholds blindly. Instrument or isolate allreduce sizes first, then screen narrower dtype/shape gates such as only FP32 scalars, or maybe a threshold below the compiler-problem shape if logs confirm a specific boundary. A useful follow-up is a guarded `VLLM_XPU_CUSTOM_ALLREDUCE_LOG_SHAPES=1` diagnostic that records `dtype`, `numel`, and `shape` with limited per-size counts.
