# MiniMax Q/K RMS Helper Retest, 2026-05-10

## Goal

Retest the default-off standalone MiniMax Q/K RMS helper after the AOT-cache
regression work. This helper is simpler than the later apply+RoPE helper: it
computes local Q/K variance from contiguous `qkv`, lets the existing vLLM TP
allreduce handle the two FP32 variance scalars, then applies Q/K RMS weights in
custom XPU kernels.

Flags:

```bash
VLLM_MINIMAX_QK_RMS_XPU_HELPER=1
VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=4
```

Model and runtime were otherwise the current MiniMax AutoRound speed path:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70
- FP16 activations
- llm-scaler raw-u4 decode-only MoE path enabled
- XPU graph disabled
- no speculative decoding
- no GPU power-limit changes

## Results

All runs used the isolated cache root:

```text
/mnt/fast-ai/vllm-cache-exp/minimax-qk-rms-helper-plain-20260510T092859Z
```

| Run | Shape | Cache state | AOT hash | GPU KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T092859Z` | p512/n512 | fresh compile | `e13662bd...` | 9,408 | 57.327 | 28.664 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T092859Z.log` |
| `20260510T093213Z` | p512/n512 | warmed AOT reload | `e13662bd...` | 17,216 | 71.444 | 35.722 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T093213Z.log` |
| `20260510T093513Z` | p512/n1536 | warmed AOT reload | `e13662bd...` | 17,216 | 48.762 | 36.572 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T093513Z.log` |

## Interpretation

The helper is functional but not a speed path. It reproduces the same cold AOT
artifact seen elsewhere: first isolated compile reports only `9,408` GPU KV
tokens and runs slowly, while a reload of the same AOT payload restores `17,216`
KV tokens and normal floor-level speed.

Even warmed, the helper trails the current default p512/n1536 floor
(`37.05` output tok/s) and the accepted MiniMax AutoRound high
(`41.130667` output tok/s). It also trails the pre-regression p512/n512 high
(`39.610585` output tok/s).

The useful lesson is that replacing the local Q/K RMS math with separate custom
var/apply kernels does not remove the expensive graph/scheduling boundary around
TP communication. It still leaves the oneCCL allreduce in the middle and appears
to create a worse compiled schedule than the stock MiniMax Q/K path.

Decision:

- keep `VLLM_MINIMAX_QK_RMS_XPU_HELPER` unset for real benchmarks;
- do not submit these runs to LocalMaxxing;
- continue toward a real graph-safe Q/K allreduce+RMS fusion rather than more
  standalone post-allreduce helpers.

