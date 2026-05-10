# MiniMax Scheduler and XCCL Microbench Screens, 2026-05-10

## Summary

After the Q/K apply+RoPE helper regressed compiled throughput, I screened two low-risk vLLM scheduling/compile settings and then measured raw XCCL allreduce latency at MiniMax decode tensor sizes.

Conclusion:

- `--no-async-scheduling` is negative.
- Static compile specialization with `--compilation-config={"compile_sizes":[1]}` is negative.
- Raw default XCCL allreduce latency is already low for the relevant tensor sizes, so the remaining issue is likely how collectives are embedded and fenced inside the decode graph rather than oneCCL's standalone algorithm selection.
- `CCL_ALLREDUCE=direct` is much slower both in the microbench and full model; keep oneCCL default `topo`.

## vLLM Screens

Baseline for p512/n512:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T011816Z.log`
- Total tok/s: `79.22117`
- Output tok/s: `39.610585`
- GPU KV cache tokens: `17,216`

`--no-async-scheduling`:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T045547Z.log`
- Total tok/s: `54.618106`
- Output tok/s: `27.309053`
- GPU KV cache tokens: `9,408`
- Result: very negative. Async scheduling should stay enabled.

Static compile size 1:

- Command extra: `--compilation-config={"compile_sizes":[1]}`
- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T045930Z.log`
- Total tok/s: `61.443969`
- Output tok/s: `30.721984`
- Compile log showed an extra compile range `(1, 1)` plus the normal `(1, 1024)`.
- Result: negative. The static decode graph does not beat the default general range on this XPU stack.

An earlier shorthand attempt with `-cc.compile_sizes+ 1` failed before model load because pydantic received `"1"` as a string. Use compact JSON when retesting compilation config.

## XCCL Microbenchmark

I extended `benchmarks/b70_xccl_allreduce_bench.py` to include MiniMax decode sizes:

- 8 bytes: Q/K variance allreduce, 2 x fp32.
- 6144 bytes: FP16 hidden allreduce, 3072 x fp16.

Default oneCCL/XCCL:

- Log: `/mnt/fast-ai/bench-results/xccl-allreduce/b70-xccl-allreduce-default-20260510T050341Z.log`
- 8 bytes: `0.016 ms`
- 6144 bytes: `0.014 ms`
- 1 MiB: `0.044 ms`
- 256 MiB: `9.627 ms`, about `27.88 GB/s`

`CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`:

- Log: `/mnt/fast-ai/bench-results/xccl-allreduce/b70-xccl-allreduce-topo-fabric-check0-20260510T050422Z.log`
- 8 bytes: `0.016 ms`
- 6144 bytes: `0.014 ms`
- Essentially identical to default, matching the full-model neutral/negative result.

`CCL_SYCL_ALLREDUCE_TMP_BUF=1`:

- Log: `/mnt/fast-ai/bench-results/xccl-allreduce/b70-xccl-allreduce-tmpbuf1-20260510T050435Z.log`
- 8 bytes: `0.016 ms`
- 6144 bytes: `0.015 ms`
- Essentially identical to default.

`CCL_ALLREDUCE_SMALL_THRESHOLD=0`:

- Log: `/mnt/fast-ai/bench-results/xccl-allreduce/b70-xccl-allreduce-smallthreshold0-20260510T050448Z.log`
- 8 bytes: `0.016 ms`
- 6144 bytes: `0.015 ms`
- Essentially identical to default.

`CCL_ALLREDUCE=direct`:

- Log: `/mnt/fast-ai/bench-results/xccl-allreduce/b70-xccl-allreduce-direct-20260510T050355Z.log`
- 8 bytes: `0.098 ms`
- 6144 bytes: `0.103 ms`
- 1 MiB: `0.680 ms`
- 256 MiB: `256.967 ms`, about `1.04 GB/s`
- Result: very negative. This agrees with Intel's oneCCL documentation that non-`topo` GPU-buffer allreduce algorithms copy data to host.

## Interpretation

The raw default XCCL latency for MiniMax-sized allreduces is lower than the synchronized per-layer timing seen inside vLLM. That means the next useful work should not be more oneCCL environment toggles. Better targets:

- reduce graph breaks/fences around TP allreduce sites;
- fuse allreduce with adjacent RMS/residual work only if the op can stay in the compiled schedule;
- investigate whether vLLM's custom distributed op wrapper adds synchronization or dispatch overhead around `xccl`;
- only attempt an XPU Lamport/IPC allreduce if we can build a reusable Level Zero IPC workspace safely.

Source for the oneCCL algorithm caveat:

- <https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-15/environment-variables.html>
