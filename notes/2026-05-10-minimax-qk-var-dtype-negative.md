# MiniMax Q/K Variance Dtype Screen, 2026-05-10

Target: `Lasimeri/MiniMax-M2.7-int4-AutoRound`, vLLM/XPU TP4 on
4x Intel Arc Pro B70, FP16 target activations, llm-scaler u4 decode MoE path,
batch 1, no speculative decode, no expert dropping, and no power-limit changes.

## Experiment

Add a default-off env gate, `VLLM_MINIMAX_QK_VAR_ALLREDUCE_DTYPE`, to cast the
two-column MiniMax Q/K RMS variance tensor from FP32 to FP16 or BF16 before the
TP allreduce, then cast back to FP32 before applying the RMS normalization.

The compiled graph confirmed that the env gate took effect for the FP16 run:
the per-layer Q/K variance collective became `f16[s72, 2]` with an explicit
cast back to FP32 after `wait_tensor`.

## Results

- Cold isolated-cache p512/n512:
  - log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qk-var-dtype/vllm-minimax-m27-autoround-tp4-p512n512-20260510T114034Z.log`
  - AOT cache: `/mnt/fast-ai/vllm-cache-exp/minimax-qk-var-fp16-20260510`
  - KV cache: `9,408` tokens
  - output throughput: `27.599251` tok/s
  - total throughput: `55.198502` tok/s
- Warm direct-load p512/n512:
  - log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qk-var-dtype/vllm-minimax-m27-autoround-tp4-p512n512-20260510T114410Z.log`
  - AOT cache direct-loaded on all four ranks
  - KV cache: `17,216` tokens
  - output throughput: `35.316180` tok/s
  - total throughput: `70.632359` tok/s

The warm result trails the current quality-conservative FP32 Q/K variance
allreduce path. It is also a precision tradeoff on a normalization statistic,
so there is no reason to promote it unless a future quality harness shows the
precision loss is harmless and a different scheduler makes it faster.

## Conclusion

Do not use `VLLM_MINIMAX_QK_VAR_ALLREDUCE_DTYPE=fp16` for real MiniMax
benchmarks. The active vLLM source and installed package were reverted to the
FP32 Q/K variance allreduce path after this screen. The patch is retained only
as a negative artifact.
