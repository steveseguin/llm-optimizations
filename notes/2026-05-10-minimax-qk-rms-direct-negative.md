# MiniMax Direct Q/K RMS Helper Negative, 2026-05-10

## Goal

Test a narrower MiniMax M2.7 attention change than the earlier graph-pass and
standalone helper attempts: avoid first materializing separate `q` and `k`
tensors in Python, compute Q/K RMS variance directly from the contiguous `qkv`
projection output with the existing `minimax_qk_rms_xpu` extension, run the
normal TP allreduce on the two FP32 variance scalars, then apply Q/K RMS weights
before RoPE.

Flag:

```bash
VLLM_MINIMAX_QK_RMS_XPU_DIRECT=1
```

Model and runtime:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70
- FP16 activations
- llm-scaler raw-u4 decode-only MoE path enabled
- `max_model_len=2048`, `max_num_batched_tokens=1024`
- XPU graph disabled
- no speculative decoding
- no GPU power-limit changes

## Results

The direct-helper runs used an isolated cache root:

```text
/mnt/fast-ai/vllm-cache/minimax-qk-direct-20260510
```

| Run | Shape | Cache state | AOT hash | GPU KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T122149Z` | p512/n512 | fresh compile | `af57de59...` | 9,408 | 53.306 | 26.653 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qk-direct/vllm-minimax-m27-autoround-tp4-p512n512-20260510T122149Z.log` |
| `20260510T122636Z` | p512/n512 | warm repeat, but AOT load failed and recompiled | `af57de59...` | 17,216 | 69.437 | 34.718 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-qk-direct/vllm-minimax-m27-autoround-tp4-p512n512-20260510T122636Z.log` |

Reference immediately before this experiment, same shape and runtime except for
the direct helper:

| Run | Shape | Cache state | AOT hash | GPU KV tokens | Total tok/s | Output tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `20260510T121708Z` | p512/n512 | loaded existing AOT | `4799a3c8...` | 17,216 | 71.640 | 35.820 |

## Interpretation

The direct helper is functional and quality-preserving in intent: it keeps the
same FP32 Q/K RMS variance reduction across TP ranks and applies the same Q/K
RMS weights before RoPE. It does not skip the Q/K allreduce.

It is still not a speed path. The first isolated compile hit the familiar cold
KV-cache artifact (`9,408` tokens) and was slow. The warm repeat recovered
normal KV-cache capacity, but only reached `34.718` output tok/s, below the
same-shape baseline of `35.820` output tok/s and below the longer-output
quality-conservative reference of `37.552538` output tok/s.

The warm repeat also printed AOT load failures:

```text
'_OpNamespace' 'minimax_qk_rms_xpu' object has no attribute 'var'
```

That means the custom op was not registered early enough for direct AOT reload,
so the run recompiled. Registering the op earlier could reduce startup
overhead, but it would not change the measured generation throughput enough to
make this helper competitive.

Decision:

- keep `VLLM_MINIMAX_QK_RMS_XPU_DIRECT` unset for real benchmarks;
- archive the patch only as `patches/vllm-minimax-qk-rms-xpu-direct-negative-20260510.patch`;
- do not submit these runs to LocalMaxxing;
- continue toward larger graph-safe fusions around TP allreduce, residual, and
  layernorm/projection boundaries rather than standalone Q/K postprocessing
  helpers.
