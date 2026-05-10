# MiniMax M2.7 Follow-ups: oneCCL, MoE Timing, and WS u4 Kernel

Date: 2026-05-09

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Runtime: vLLM/XPU TP4 on 4x Arc Pro B70, `USE_LLM_SCALER_MOE=1`, `CCL_IPC=default`, `DTYPE=float16`, `MAX_MODEL_LEN=2048`, `MAX_BATCHED_TOKENS=1024`, `MAX_NUM_SEQS=1`.

## oneCCL Small-Payload Follow-up

The standalone MiniMax hidden-size XCCL allreduce probe still shows that raw collectives are not the primary ceiling:

- baseline `CCL_ZE_IPC_EXCHANGE=pidfd`, hidden fp16/fp32 payloads: about `0.0146-0.0150 ms`;
- `CCL_MAX_SHORT_SIZE=65536` and `1048576`: within noise or slightly slower;
- `CCL_PRIORITY=lifo`: slightly slower;
- `CCL_WORKER_COUNT=1`: within noise.

Logs:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/xccl-hidden-allreduce-pidfd-baseline-20260509T180129Z.log`
- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/xccl-hidden-allreduce-variants-20260509T180149Z.log`

Conclusion: keep the current vLLM default IPC path for model runs, and keep explicit `pidfd` only for standalone XCCL probes. The next speed work should stay in model execution rather than oneCCL env tuning.

## oneCCL Direct Algorithm Follow-up

On 2026-05-10 I also tested forcing the oneCCL allreduce algorithm:

```bash
CCL_ALLREDUCE=direct
```

with the current fast-NVMe MiniMax TP4 p512/n512 harness. It stalled during distributed/CCL initialization before shard loading completed and was terminated by the benchmark timeout/cleanup path. This is worse than the default algorithm selection and is not a serving candidate.

Log:

- `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T022446Z.log`

Conclusion: do not force `CCL_ALLREDUCE=direct` for MiniMax TP4 on the current B70 stack. Keep oneCCL algorithm selection on default unless a lower-level microbenchmark shows a specific model-run-safe algorithm win.

## XPU Async-Wait Allreduce Hook

I added a default-off diagnostic hook in `XpuCommunicator.all_reduce`:

```text
VLLM_XPU_ALLREDUCE_ASYNC_WAIT=1
```

It calls `dist.all_reduce(..., async_op=True)` and immediately waits, so the math is unchanged. The compiled MiniMax TP4 path cannot use it: TorchDynamo rejects `async_op=True` for distributed collectives during AOT capture. The first p512/n512 attempt failed before benchmarking with:

```text
torch._dynamo.exc.Unsupported: async_op=True for distributed collectives
```

I tightened the guard so the hook is ignored while `torch.compiler.is_compiling()`, then verified a tiny compiled p1/n8 smoke completes with the env var set. That means the patch is safe to keep as an eager-only diagnostic, but it is not a current performance path.

I also ran a full guarded BF16 0.95 p512/n512 pass after the smoke. It completed, but because the hook is ignored inside compiled collectives it behaved like the normal compiled path:

- p512/n512, BF16, `--gpu-memory-utilization 0.95`: `35.95` output tok/s, `71.90` total tok/s;
- KV cache: `18,880` tokens;
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T030708Z.log`.

This is neutral-to-negative versus the BF16 p512/n512 and p512/n1024 baselines, so it is not a LocalMaxxing submission.

Artifacts:

- patch: `patches/vllm-xpu-allreduce-async-wait-guard-20260510.patch`
- blocked compiled p512/n512 log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T024748Z.log`
- guarded p1/n8 smoke log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260510T025024Z.log`
- guarded p512/n512 full log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T030708Z.log`

Conclusion: do not use async-op allreduce as a compiled MiniMax optimization unless we move the collective behind a graph-safe custom op or alter the graph partitioning so Dynamo does not trace `async_op=True`.

## vLLM MoE Timing Split

A temporary `MoERunner` timing patch separated router expert selection from quant apply on a short p512/n4 diagnostic:

- `moe.select_experts`: about `0.058-0.064 ms/layer` during decode;
- `moe.llm_scaler_u4_bridge`: about `0.098-0.106 ms/layer`;
- `moe.quant_apply`: about `0.176-0.185 ms/layer` around the bridge in decode;
- prefill/warmup still uses `moe.fused_experts_fallback`, about `2.4-3.0 ms/layer` in this diagnostic.

Log:

- `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n4-20260509T180217Z.log`

Conclusion: router top-k is not the next best target. The remaining MoE-side target is the routed expert apply path and its surrounding launch/bridge behavior. The temporary `MoERunner` timing wrappers were removed from the active runtime after measurement.

## Direct Torch Op Dispatch Screen

I tested replacing the Python wrapper around `moe_forward_tiny_cutlass_nmajor_int4_u4` with a cached direct `torch.ops.moe_int4_ops` call.

Standalone fake MiniMax layer:

- wrapper: `0.085629 ms`;
- direct op: `0.079450 ms`.

Real p512/n256 model run:

- direct-op patch: `33.622344 output tok/s`;
- restored scalar path control: `33.609758 output tok/s`;
- current best remains `34.578045 output tok/s` for p512/n256 and `37.136187 output tok/s` for p512/n512.

Conclusion: direct dispatch is too small/noisy in the full model. Do not promote.

## ESIMD Work-Sharing u4 MoE Kernel

I added an experimental unsigned u4 no-shared ESIMD work-sharing kernel:

- `moe_forward_tiny_cutlass_nmajor_int4_u4_ws`
- source: `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/csrc/moe_batch/moe_int4.sycl`
- Python export: `/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/ops.py`

Standalone fake MiniMax-shaped layer looked promising:

- `m=1`: scalar u4 `0.174667 ms`, WS u4 `0.030994 ms`, max abs diff `6.10352e-05`;
- `m=4`: scalar u4 `0.295609 ms`, WS u4 `0.121688 ms`, max abs diff `0.00195312`.

Real model validation did not carry that win:

- p512/n256 WS: `31.654977 output tok/s`, `94.964932 total tok/s`;
- p512/n512 WS: `35.389129 output tok/s`, `70.778259 total tok/s`;
- both are below the scalar u4 best.

Logs:

- build: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/build-moe-int4-u4-ws-20260509T183222Z.log`
- p512/n256: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260509T183428Z.log`
- p512/n512: `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260509T184219Z.log`

Conclusion: keep the active vLLM path on scalar `moe_forward_tiny_cutlass_nmajor_int4_u4`. The WS kernel is a useful negative: a faster isolated kernel can still regress the four-rank model run, likely due scheduling, cache behavior, or graph/worker interactions not represented by the single-GPU microbench.

## LocalMaxxing

No new LocalMaxxing submission from this round. These are negative/diagnostic results, while the current public best for this model remains the previously submitted scalar u4 p512/n512 run, `cmoyagit0004dmk014gk25e2k`.

## Next Work

- prioritize an XPU equivalent of the CUDA MiniMax Q/K allreduce+RMS fusion, since Q/K norm plus TP collectives remain high-value and quality-preserving;
- if returning to MoE kernels, target a true fused up/down route kernel or persistent decode path rather than the ESIMD WS split tested here;
- continue treating n-gram/speculative decode as negative for this MiniMax checkpoint until there is a real MiniMax draft/MTP path.
