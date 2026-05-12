# MiniMax Static Compile Follow-Ups, 2026-05-12

These tests followed the `compile_sizes=[1]` win for MiniMax M2.7 AutoRound
W4A16 on four Arc Pro B70 GPUs. The baseline to beat remains:

- `47.586110` output tok/s, `63.448146` total tok/s at p512/n1536.
- TP4, FP16 activations, AutoRound INT4 W4A16 weights.
- `--compilation-config '{"use_inductor_graph_partition":true,"compile_sizes":[1]}'`
- AOT `3e2cefa134c3aecc743c56d36960e4cb0a8ac7d2adc73c3f2a078cc8b6164846`.
- No speculation, no expert dropping, no power-limit change.

## Results

| Test | Prompt/Output | KV tokens | Output tok/s | Total tok/s | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| `max_num_batched_tokens=512`, compile `[1]` | 512/512 | 9,472 | 32.95 | 65.90 | Negative, low-KV artifact |
| compile `[1,512]`, combo kernels | 512/512 | n/a | n/a | n/a | Failed in Inductor/IGC |
| compile `[1,512]`, no combo | 512/512 | n/a | n/a | n/a | Failed KV check at 0.08 GiB free |
| compile `[1,512]`, no combo, `gpu_memory_utilization=0.95` | 512/512 | 32,960 | 30.40 | 60.80 | Negative |
| compile `[1]`, `gpu_memory_utilization=0.95` | 512/1536 | 33,024 | 46.86 | 62.48 | Useful context headroom, slight speed loss |
| compile `[1]`, `compile_ranges_endpoints=[512]` | 512/512 | 9,408 | 31.88 | 63.75 | Negative, low-KV artifact |
| compile `[1]`, no combo kernels | 512/512 | 9,408 | 32.58 | 65.15 | Negative, combo kernels are part of win |
| forced XPU graph, no no-op communicator guard | 64/32 | n/a | n/a | n/a | Failed `CudaCommunicator` assert |
| forced XPU graph with no-op communicator guard | 64/32 | 9,472 | 4.94 | 14.82 | Runs but unusably slow |
| compile `[1]`, gmem 0.95, configured 32k context | 512/1536 | n/a | n/a | n/a | Failed KV check after graph memory overhead |
| compile `[1]`, gmem 0.95, configured 24k context | 512/512 | 25,600 | 33.81 | 67.61 | Works as context-capacity proof, LocalMaxxing `cmp2p5jhb009wrm01cmkurcfa` |

## Interpretation

The win is narrow and specific: graph partition plus a static decode-size graph
for one token. Extra compile ranges, static prefill specialization, and disabled
combo kernels all shift vLLM back into the bad low-KV shape or spend so much
memory on graph artifacts that throughput collapses.

`gpu_memory_utilization=0.95` is worth keeping as a context-capacity knob. On
the default 2k configured window it doubled the reported KV budget from roughly
`16k` to `33k` tokens while only dropping p512/n1536 decode from `47.586` to
`46.86` tok/s. That number does not translate directly into a working 32k
configured context once compile graph memory is reserved: configured 32k failed
with `1.52 GiB` free for KV versus `1.94 GiB` required. Configured 24k did work,
with `25,600` KV tokens and `33.81` output tok/s at p512/n512. Treat 24k as the
current high-context proof point, not a speed result. LocalMaxxing accepted this
capacity datapoint as `cmp2p5jhb009wrm01cmkurcfa`.

The XPU graph path remains closed for throughput. The existing local
`VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1` patch gets past the CUDA-only communicator
assert, but PIECEWISE graph capture still falls into the low-KV artifact and
only reaches `4.94` output tok/s on a small p64/n32 smoke.

## Compiler Bug Notes

Static prefill and static decode variants repeatedly trigger Intel `ocloc` /
IGC failures such as:

```text
IGC: Internal Compiler Error: Floating point exception
triton_per_fused__to_copy_mean_pow_split_with_sizes_3
triton_red_fused__to_copy_mm_t_9
```

The decode-only winning path can recover from these errors and load AOT on warm
runs. Static prefill plus combo kernels can hard-fail during compile. Static
prefill without combo kernels can compile, but memory and throughput are bad.

## Logs

- `max_num_batched_tokens=512`: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-maxtok512-p512n512-20260512T124902Z`
- compile `[1,512]` combo failure: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-512-p512n512-20260512T125510Z`
- compile `[1,512]` no-combo failure: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-512-nocombo-p512n512-20260512T125918Z`
- compile `[1,512]` no-combo gmem 0.95: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-512-nocombo-gmem095-p512n512-20260512T130704Z`
- compile `[1]` gmem 0.95: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-gmem095-p512n1536-20260512T131007Z`
- compile range split 512: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-range512-p512n512-20260512T131335Z`
- compile `[1]` no combo: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-nocombo-p512n512-20260512T132125Z`
- XPU graph assert: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-xpugraph-force-p64n32-20260512T132742Z`
- XPU graph no-op communicator: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-xpugraph-noopcomm-p64n32-20260512T133404Z`
- configured 32k context KV failure: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-gmem095-ctx32768-p512n1536-20260512T134125Z`
- configured 24k context proof: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-inductor-partition-compile1-gmem095-ctx24576-p512n512-20260512T134732Z`
