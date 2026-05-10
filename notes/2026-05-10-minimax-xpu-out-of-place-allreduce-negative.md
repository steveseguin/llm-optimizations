# MiniMax XPU Out-Of-Place Allreduce Screen

Date: 2026-05-10

## Hypothesis

The compiled XPU TP allreduce path cloned every collective input, waited on the
functional allreduce, then copied the result back into the clone. For the
quality-conservative MiniMax M2.7 AutoRound TP4 graph this meant 187 collective
sites also carried 187 clone/copy pairs.

Because vLLM consumes the returned allreduce tensor, a guarded out-of-place
functional allreduce looked like a possible way to remove those clone/copy
pairs without changing math:

```text
VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE=1
```

Patch archived at:

```text
patches/vllm-xpu-compile-out-of-place-allreduce-experiment-20260510.patch
```

## Implementation

The guarded runtime patch changes only the compile-time XPU allreduce path:

```python
if (
    os.environ.get("VLLM_XPU_COMPILE_OUT_OF_PLACE_ALLREDUCE", "0") == "1"
    and torch.compiler.is_compiling()
):
    return funcol.all_reduce(input_, "sum", self.device_group)
```

The default path remains unchanged when the env flag is unset.

## Graph Result

Smoke run:

- shape: p64/n32
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oop-ar-smoke/vllm-minimax-m27-autoround-tp4-p64n32-20260510T174921Z.log`
- cache: `/mnt/fast-ai/vllm-cache/minimax-m2.7-autoround-oop-ar-smoke2-20260510T174921Z/torch_compile_cache/8b9fd5d36a/rank_0_0/backbone`

The final implementation produced:

| Graph metric | Count |
| --- | ---: |
| `_c10d_functional.all_reduce` call lines | 187 |
| actual `wait_tensor` call lines | 187 |
| allreduce-adjacent `.clone()` / `copy_()` occurrences | 0 |

So the graph transformation worked structurally.

An earlier version called `funcol.wait_tensor(...)` inside the communicator.
That also generated, but it produced two waits per collective and was abandoned.

## Throughput Result

Quality-conservative p512/n512 screens with the flag enabled:

| Run | Total tok/s | Output tok/s | GPU KV tokens | Log |
| --- | ---: | ---: | ---: | --- |
| cold/default cache | `72.566492` | `36.28` | 13,632 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oop-ar/vllm-minimax-m27-autoround-tp4-p512n512-20260510T175301Z.log` |
| warm/default cache | `71.285115` | `35.64` | 17,216 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-oop-ar/vllm-minimax-m27-autoround-tp4-p512n512-20260510T175607Z.log` |

Accepted quality-conservative p512/n512 reference is `39.610585` output tok/s.
This patch is therefore a regression even when KV headroom returns to 17,216
tokens.

## Conclusion

Do not promote this patch.

Removing clone/copy from the Python-visible compiled graph is not enough. The
out-of-place functional collective likely schedules worse than the current
clone/in-place-style path, or the clone/copy work was not a material part of the
wall-clock bottleneck. The next viable path still needs a real fused collective
epilogue, not just a cleaner functional allreduce trace.

No LocalMaxxing submission: this is a negative internal optimization screen.
