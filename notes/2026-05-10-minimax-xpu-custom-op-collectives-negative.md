# MiniMax XPU Custom-Op Collectives Screen, 2026-05-10

## Goal

Test whether opting XPU into vLLM's opaque collective custom-op path can remove
the visible `_c10d_functional.all_reduce` / `wait_tensor` graph boundaries that
show up in the current MiniMax M2.7 TP4 AOT graph.

Patch:

```bash
VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1
```

adds a default-off `XPUPlatform.use_custom_op_collectives()` override. It does
not enable CUDA custom allreduce; it only routes tensor-parallel collectives
through `torch.ops.vllm.all_reduce`.

## Results

All runs used:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- 4x Intel Arc Pro B70, TP4
- vLLM 0.20.1-local / XPU / FP16
- llm-scaler unsigned-u4 MiniMax MoE decode path
- no speculation, no expert dropping, no Q/K allreduce skip, no power changes

| Run | Shape | Cache | KV tokens | Total tok/s | Output tok/s | Log |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `20260510T180956Z` | p64/n32 | cold | 9,472 | 18.993 | 6.331 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-customop-collectives/vllm-minimax-m27-autoround-tp4-p64n32-20260510T180956Z.log` |
| `20260510T181307Z` | p64/n32 | warm | n/a | 74.913 | 24.971 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-customop-collectives/vllm-minimax-m27-autoround-tp4-p64n32-20260510T181307Z.log` |
| `20260510T181527Z` | p512/n512 | cold | 9,408 | 53.720 | 26.860 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-customop-collectives/vllm-minimax-m27-autoround-tp4-p512n512-20260510T181527Z.log` |
| `20260510T181909Z` | p512/n512 | warm | 17,280 | 69.960 | 34.980 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-xpu-customop-collectives/vllm-minimax-m27-autoround-tp4-p512n512-20260510T181909Z.log` |

Reference points:

- current clean p512/n512 refresh: `35.648` output tok/s;
- earlier accepted p512/n512 high: `39.611` output tok/s.

## Graph Census

The p512/n512 AOT graph did change:

```text
all_reduce_comment_lines=40
all_reduce_call_lines=0
wait_tensor_call_lines=0
rms_int4_lines=16
fused_add_rms_lines=0
```

The source comments identify those collective sites as `vllm.all_reduce`
instead of `_c10d_functional.all_reduce`. So the experiment successfully
removed the explicit compiled oneCCL wait nodes.

## Interpretation

This is still negative. The opaque custom-op path emits PyTorch aliasing
warnings because the XPU allreduce returns the input tensor on the eager side:

```text
vllm::all_reduce ... output ... must not also be an input ...
```

Even after AOT reload restores the normal KV-cache size, the run is slightly
slower than the current clean refresh and materially slower than the older
p512/n512 high. Removing visible `wait_tensor` nodes is not enough when the
collective is still a Python-level opaque runtime boundary.

Decision:

- keep `VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES` unset for real MiniMax benchmarks;
- do not submit these runs to LocalMaxxing;
- continue toward C++/SYCL or backend-level collective fusion, especially Q/K
  variance allreduce+RMS and hidden-state allreduce+residual/RMS epilogues.

