# MiniMax U4 Logits MoE Screen, 2026-05-10

## Purpose

Test whether the llm-scaler INT4 MoE kernel that accepts router logits directly
can reduce MiniMax M2.7 decode overhead by folding top-k selection into the MoE
call. This is a quality-preserving candidate because it still routes through
the same target model logits and experts; it is not speculation or expert
dropping.

## Build

I rebuilt a logits-capable MoE-only extension in a separate work copy so the
known-good active extension was not replaced:

```text
/home/steve/src/llm-scaler-u4-logits/vllm/custom-esimd-kernels-vllm
```

Build recipe:

```bash
set +u
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
set -u
source /home/steve/.venvs/vllm-xpu/bin/activate
export LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib:${LD_LIBRARY_PATH:-}
cd /home/steve/src/llm-scaler-u4-logits/vllm/custom-esimd-kernels-vllm
python setup_moe_int4_only.py build_ext --inplace
```

The resulting extension imports cleanly with venv libs first and exports:

- `moe_forward_tiny_cutlass_nmajor_int4_u4`
- `moe_forward_tiny_cutlass_nmajor_int4_u4_logits`
- `moe_forward_tiny_cutlass_nmajor_int4_u4_ws`

## vLLM Hook

The active vLLM runtime now has a default-off env gate:

```text
VLLM_XPU_USE_LLM_SCALER_MOE_LOGITS=1
VLLM_XPU_LLM_SCALER_MOE_LOGITS_MAX_TOKENS=<N>
```

Patch artifact:

```text
patches/vllm-minimax-u4-logits-moe-screen-20260510.patch
```

Default behavior is unchanged because `LOGITS_MAX_TOKENS` defaults to `4` and
the env flag must be explicitly enabled.

The hook also forces `x` and `router_logits` contiguous before calling the
kernel. Without that, the first forced screen failed during worker warmup with:

```text
Expected x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 64 && x.is_contiguous() to be true
```

## Results

All runs were TP4 on 4x B70, FP16 activation path, `max_model_len=512`,
`max_num_batched_tokens=256`, p64/n32.

| Variant | Total tok/s | Output tok/s | Status | Log |
| --- | ---: | ---: | --- | --- |
| clean pre-screen | `79.597017` | `26.532339` | healthy short baseline | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-u4-logits-smoke-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T205119Z.log` |
| logits env on, default max `4` | `75.713504` | `25.237835` | branch not visibly exercised; no speedup | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-u4-logits-smoke-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T205401Z.log` |
| logits forced max `256` | none | none | failed warmup; kernel contract is `M <= 64` and contiguous | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-u4-logits-max256-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T205711Z.log` |
| logits forced max `64` plus contiguous inputs | `19.083764` | `6.361255` | completed but severe regression | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-u4-logits-max64-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T205948Z.log` |
| post-screen clean default | `75.616687` | `25.205562` | default path still healthy | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-post-logits-clean-smoke-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T210256Z.log` |

## Interpretation

The logits-capable kernel is not a promotion path as-is. It is reachable and
imports cleanly, but when forced into the compiled p64/n32 path it becomes much
slower than the raw U4 top-k-weight path. The most likely causes are:

- the kernel has an `M <= 64` tiny contract and is poorly matched to vLLM's
  profiling/compile range;
- folding top-k into the MoE call does not remove the larger TP collective
  bottleneck;
- the extra contiguity and compile graph shape cost dominates any top-k savings.

Decision:

- keep the logits hook default-off;
- do not replace the active known-good MoE extension with the logits build;
- do not submit these runs to LocalMaxxing;
- revisit only if we add a decode-only graph path that guarantees `M <= 64`
  without contaminating prefill/warmup, or if we modify the llm-scaler kernel to
  handle the exact vLLM symbolic shape more efficiently.

## Side Finding

The earlier attempted "combined MoE-delay/post-attention" screen did not
exercise those historical env hooks in the currently installed runtime. Those
runs should be treated as mislabeled clean p512/n512 baseline repeats, not as
evidence for or against the old delayed-allreduce patches.
