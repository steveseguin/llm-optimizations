# MiniMax Restore, Long Output, and htile Negative

## Summary

After the reboot and htile experiment, the active MiniMax AutoRound path was
functional but slower than the earlier fast-NVMe baseline. The custom
llm-scaler u4 op was still registered correctly, but vLLM had moved from the
known-good `c15860...` AOT graph cache to `c952f...`.

Removing the default-off Q/K apply+RoPE helper branch from
`vllm/model_executor/models/minimax_m2.py` restored the `c15860...` AOT cache
selection and recovered most of the lost speed:

| Shape | Before restore | After restore | Notes |
| --- | ---: | ---: | --- |
| p512/n512 | `36.012` output tok/s | `38.998` output tok/s | restored `c15860...` AOT cache |
| p512/n1536 | `36.821` output tok/s | `39.450`, then `39.961` output tok/s | still below the 41.131 best, but back in the fast path band |

Patch artifact:

- `patches/vllm-minimax-remove-qk-apply-rope-branch-restore-c158-20260510.patch`

The previous apply+RoPE helper remains documented as a negative experiment. It
was numerically valid, but even as a default-off branch it changed the compiled
graph cache choice and cost several output tok/s on the active stack.

## Runtime Verification

The restored llm-scaler u4 custom op was checked before retesting:

- `torch.ops.moe_int4_ops.moe_forward_tiny_cutlass_nmajor_int4_u4` is visible.
- A small vLLM smoke, p128/n32, completed with `120.178` total tok/s.
- The active vLLM Python file passes `py_compile` after branch removal.

Relevant logs:

- Smoke: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p128n32-20260510T053730Z.log`
- Pre-restore p512/n1536: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T053933Z.log`
- Restored p512/n512: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T054939Z.log`
- Restored p512/n1536: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T055206Z.log`
- Restored p512/n1536 repeat: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T055509Z.log`

No new LocalMaxxing result was submitted for these restore checks because the
existing accepted `41.130667` p512/n1536 result remains faster.

## Long Output

I also screened p512/n2048:

| Config | Output tok/s | Total tok/s | KV cache tokens | Interpretation |
| --- | ---: | ---: | ---: | --- |
| default memory, max len 2048 | `33.925` | `42.406` | `9,408` | valid, but slower than p512/n1536 |
| `--gpu-memory-utilization 0.95`, max len 2048 | `36.772` | `45.964` | `33,408` | capacity improves, speed still below best |

The 0.95 setting is useful for capacity and cache headroom. It is not a raw
speed path for this batch-1 MiniMax harness.

## XCCL Out-of-Place Microbench

I extended `benchmarks/b70_xccl_allreduce_bench.py` with out-of-place modes to
mimic vLLM's `output = input_.clone()` allreduce path. Results on 4x B70 with
default oneCCL/XCCL:

| Mode | 8 B | 6,144 B | 1 MiB | 256 MiB |
| --- | ---: | ---: | ---: | ---: |
| in-place | `0.016 ms` | `0.015 ms` | `0.042 ms` | `9.636 ms` |
| clone then allreduce | `0.021 ms` | `0.020 ms` | `0.044 ms` | `10.667 ms` |
| empty copy then allreduce | `0.021 ms` | `0.020 ms` | `0.044 ms` | `10.667 ms` |

The clone/copy tax is about `0.005 ms` for tiny decode allreduces. That is real,
but not enough to explain the MiniMax ceiling by itself. The remaining issue is
still likely graph/fence placement around collectives and attention/projection
work rather than raw oneCCL latency.

## htile Negative

The llm-scaler down-projection htile experiment looked promising in isolation:

- Synthetic MiniMax-shape down kernel median improved from `140.425 us` to
  `44.430 us`.
- Synthetic output matched exactly: max diff `0.0`, mean diff `0.0`.

Full vLLM throughput was negative:

- htile p512/n512: `35.067` output tok/s.
- control after that rebuild: `36.012` output tok/s.
- restored active path after removing the apply+RoPE branch: `38.998` output
  tok/s.

The htile branch should not be promoted. The patch is saved as:

- `patches/llm-scaler-minimax-u4-down-htile-negative-20260510.patch`

Important caution: that patch was captured against upstream llm-scaler and
includes the prior u4 MiniMax work plus the failed htile addition. Do not
reverse-apply it over the active runtime, because it will remove required u4
custom-op registrations too.

## Next Work

- Keep `VLLM_MINIMAX_QK_APPLY_ROPE_XPU_HELPER` absent and leave
  `VLLM_MINIMAX_QK_RMS_XPU_HELPER` unset for benchmark runs.
- Treat the active MiniMax speed path as: vLLM/XPU TP4, FP16 activations,
  llm-scaler unsigned-u4 MoE decode bridge, default oneCCL, XPU graph disabled,
  max len 2048 for speed mode.
- Next useful source work should target compiled graph/fence placement around
  Q/K allreduce and attention/projection, not standalone oneCCL environment
  toggles or standalone RMS helper kernels.
