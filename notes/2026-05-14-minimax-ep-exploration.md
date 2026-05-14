# MiniMax Expert Parallel Exploration

Goal: test whether MiniMax M2.7 AutoRound W4A16 can exceed the TP4-only
`61.08` to `61.75` output tok/s range by enabling vLLM expert parallelism on
four Arc Pro B70 GPUs.

## Baseline For Comparison

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Runtime: vLLM `0.20.1-local c51df4300`, Intel XPU
- Baseline recipe: TP4, no EP, llm-scaler INT4 MoE decode, TRITON_ATTN,
  full-decode-only XPU graph
- Fresh gated repeat: `61.0808` mean output tok/s, `81.4411` mean total tok/s
- Best prior gated repeat: about `61.75` output tok/s

## EP Attempt 1: Stock Config Lookup

Command shape:

```bash
vllm bench throughput \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --random-input-len 512 \
  --random-output-len 1536 \
  --max-model-len 2048 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --async-engine \
  --block-size 256 \
  --attention-backend TRITON_ATTN \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_num_of_warmups":0,"compile_sizes":[1]}'
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T111224Z.log`
- EP did activate: workers were named `Worker_TP0_EP0` through
  `Worker_TP3_EP3`.
- vLLM reported local/global experts as `64/256`.
- The MoE shape changed from the tuned TP-only shape `E=256,N=384` to EP-local
  shape `E=64,N=1536`.
- No tuned config existed for
  `E=64,N=1536,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json`.
- Runtime fell back to default MoE config and then repeatedly reported:
  `No available shared memory broadcast block found in 60 seconds`.
- Run was killed. No benchmark datapoint accepted.

## EP Attempt 2: Add Local E=64,N=1536 Config

Added a first-pass config at:

`configs/moe/E=64,N=1536,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json`

The config mirrors the known-good Intel B70 decode settings for `E=256,N=384`
and is supplied through:

```bash
VLLM_TUNED_CONFIG_FOLDER=/home/steve/llm-optimizations-publish/configs/moe
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n256-20260514T111818Z.log`
- Shorter `512/256` probe still stalled before weight loading reached the
  normal high-VRAM stage.
- GPU memory stayed below 1 GiB per card while the process sat in distributed
  initialization/worker setup.
- Run was killed. No benchmark datapoint accepted.

## EP Attempt 3: Eager, No XPU Graph

Command shape:

```bash
VLLM_TUNED_CONFIG_FOLDER=/home/steve/llm-optimizations-publish/configs/moe \
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1 \
TP=4 \
XPU_GRAPH=0 \
USE_LLM_SCALER_MOE=1 \
MAX_MODEL_LEN=2048 \
MAX_BATCHED_TOKENS=512 \
MAX_NUM_SEQS=1 \
INPUT_LEN=512 \
OUTPUT_LEN=128 \
DTYPE=float16 \
EXTRA_ARGS='--enforce-eager --block-size 256 --no-enable-prefix-caching --attention-backend TRITON_ATTN --enable-expert-parallel --all2all-backend allgather_reducescatter --compilation-config {"mode":0,"cudagraph_mode":"NONE","cudagraph_num_of_warmups":0,"compile_sizes":[1]}' \
  scripts/bench-vllm-minimax-autoround-xpu.sh
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n128-20260514T113821Z.log`
- EP reached full worker/model load with `TP0_EP0` through `TP3_EP3`.
- It still stalled with repeated
  `No available shared memory broadcast block found in 60 seconds`.
- This means the first EP blocker is not only XPU graph capture.
- Run was killed. No benchmark datapoint accepted.

## EP Attempt 4: Requested Naive All-to-All

Command shape was the same as Attempt 3 but with:

```bash
--all2all-backend naive
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n128-20260514T114208Z.log`
- This run got further than Attempt 3: model load, local `E=64,N=1536`
  config pickup, KV cache allocation, and prompt processing.
- It then failed during decode/sampling in the async output path with:
  `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`.
- The relevant stack was in `WorkerAsyncOutputCopy` /
  `gpu_model_runner.py` / `update_async_output_token_ids`.
- Run was killed. No benchmark datapoint accepted.

## EP Attempt 5: Requested Naive All-to-All, Async Scheduling Disabled

Command shape was Attempt 4 plus:

```bash
--no-async-scheduling
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n128-20260514T114743Z.log`
- vLLM reported:
  `The 'naive' all2all backend has been removed. Falling back to 'allgather_reducescatter'.`
- It confirmed `Asynchronous scheduling is disabled.`
- The run stalled in distributed initialization/worker setup at low VRAM
  instead of reaching model load or decode.
- GPU memory stayed around 0.9 to 1.1 GiB per card before termination.
- Run was killed. No benchmark datapoint accepted.

## Current EP Read

EP is still worth pursuing, but it is not a drop-in speed win on this stack yet.
The current blockers are:

- EP/all-to-all initialization is unreliable on the current XPU+oneCCL path.
- Disabling XPU graph does not clear the stall.
- Disabling async scheduling does not clear the stall.
- `--all2all-backend naive` is not a true control on this vLLM build because
  vLLM falls back to `allgather_reducescatter`.
- The EP-local MoE shape needs a real tuned XPU config and probably kernel
  validation, not just copied TP-only parameters.
- The available all-to-all backend used here,
  `allgather_reducescatter`, may be too expensive or unstable on four PCIe B70s.

Do not submit these EP attempts to LocalMaxxing; they are failure/diagnostic
results only.

## Source Inspection And Synthetic Ag/Rs Repro

Source read:

- `vllm/config/parallel.py` still accepts `naive` in the type list, but the
  validator rewrites both `pplx` and `naive` to `allgather_reducescatter`.
- `vllm/distributed/device_communicators/xpu_communicator.py` only wires XPU EP
  all-to-all to `AgRsAll2AllManager`.
- In practice, there is no real non-AG/RS XPU EP all-to-all backend in this
  local vLLM tree.

Added:

- `benchmarks/b70_xccl_ag_rs_bench.py`

Equal-size Ag/Rs microbench:

- Log:
  `/home/steve/bench-results/xccl-ag-rs/b70-xccl-ag-rs-vllm-compat-20260514T115640Z.log`
- Runtime:
  `torchrun --standalone --nproc_per_node=4 benchmarks/b70_xccl_ag_rs_bench.py`
- Environment:
  `CCL_ATL_TRANSPORT=ofi`, `CCL_TOPO_P2P_ACCESS=1`,
  `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3`,
  `ZE_AFFINITY_MASK=0,1,2,3`
- The equal-size vLLM-compatible gather branch completed.
- Decode-sized `tokens=1, hidden=6144` timings:
  - `all_gather_into_tensor_equal`: `0.0215 ms`
  - `all_gather_list_equal`: `0.0678 ms`
  - `all_gather_vllm_xpu_equal`: `0.0250 ms`
  - `reduce_scatter_tensor_equal`: `0.0179 ms`
  - `reduce_scatter_list_equal`: `0.0426 ms`
- Larger `tokens=512, hidden=6144` timings:
  - `all_gather_into_tensor_equal`: `0.4601 ms`
  - `all_gather_vllm_xpu_equal`: `0.5490 ms`
  - `reduce_scatter_tensor_equal`: `0.4593 ms`
  - `reduce_scatter_list_equal`: `0.5430 ms`

Uneven-size Ag/Rs repro:

- Log:
  `/home/steve/bench-results/xccl-ag-rs/b70-xccl-ag-rs-uneven-timeout-20260514T115922Z.log`
- Command added `B70_AGRS_INCLUDE_UNEVEN=1`.
- Equal-size cases finished, then the first uneven `all_gather` did not return
  before the 60-second timeout.
- GPUs returned idle after timeout.

Padded uneven-size Ag/Rs repro:

- Log:
  `/home/steve/bench-results/xccl-ag-rs/b70-xccl-ag-rs-padded-uneven-20260514T120324Z.log`
- Added a padded uneven gather that pads all ranks to `max(sizes)`, gathers with
  `all_gather_into_tensor`, then slices back to the requested per-rank sizes.
- The padded uneven gather completed:
  `all_gather_padded_uneven,1,6144,20,0.0602,0.1172`
- The following raw uneven `all_gather(list, input)` still hung and was killed
  by timeout.

Interpretation:

- Equal-size AG/RS collectives are not the root EP blocker by themselves.
- The variable-size `dist.all_gather` path used by vLLM-style `all_gatherv`
  appears unsafe on this XPU/XCCL stack.
- A plausible next patch is to avoid uneven `all_gatherv` for XPU EP by padding
  per-rank token chunks to an equal size before gather, then slicing after
  dispatch/combine. This may trade a little bandwidth for correctness and
  avoids another full-model blind launch.

## EP Attempt 6: Padded XPU Allgatherv, Async Scheduling On

Local source patch:

- `vllm/distributed/device_communicators/xpu_communicator.py`
- Patch snapshot:
  `patches/vllm-xpu-padded-allgatherv-sync-output-20260514.patch`
- Guard: `VLLM_XPU_PAD_UNEVEN_ALLGATHERV=1`
- Behavior: for uneven `all_gatherv`, pad to equal-size chunks and use
  `dist.all_gather_into_tensor`, then slice.

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n64-20260514T120449Z.log`
- EP reached model load, KV allocation, prompt processing, and decode/sampling.
- It then hit the same async output-copy fault:
  `RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`.
- Relevant stack:
  `WorkerAsyncOutputCopy` / `gpu_model_runner.py` /
  `gpu_input_batch.py:update_async_output_token_ids`.
- No benchmark datapoint accepted.

## EP Attempt 7: Padded XPU Allgatherv, Async Scheduling Off

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n64-20260514T120748Z.log`
- Command added `--no-async-scheduling`.
- The run stalled in distributed initialization/worker setup at low VRAM before
  model load.
- No benchmark datapoint accepted.

## EP Attempt 8: Padded XPU Allgatherv, Sync XPU Output Copy

Local source patch:

- `vllm/v1/worker/gpu_model_runner.py`
- Patch snapshot:
  `patches/vllm-xpu-padded-allgatherv-sync-output-20260514.patch`
- Guard: `VLLM_XPU_SYNC_ASYNC_OUTPUT_COPY=1`
- Behavior: keep async scheduling enabled, but for XPU copy sampled token ids to
  CPU synchronously and use a completed no-op event instead of the async copy
  stream/event.

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n64-20260514T121053Z.log`
- EP reached full model load.
- It did not throw the async output-copy `DEVICE_LOST` error before being
  stopped.
- It then hung after model load and emitted:
  `No available shared memory broadcast block found in 60 seconds`.
- No benchmark datapoint accepted.

Current read after Attempts 6-8:

- The padded allgatherv patch fixes a real synthetic XCCL hang and gets the
  full EP path past one prior hazard.
- EP still is not stable enough to benchmark.
- Async-on reaches decode but loses the device in output-copy/sampling.
- Async-off avoids the output-copy path but can stall during initialization.
- Sync XPU output copy avoids the observed output-copy exception but exposes a
  later worker/shared-memory synchronization stall after model load.

## Next EP Work

- Stop treating `naive` as available unless the current vLLM all-to-all registry
  is patched or an older backend is restored.
- Keep the XPU-only padded equal-size `all_gatherv` patch as a candidate fix,
  but do not promote it until the full EP path can complete a correctness run.
- Check whether MiniMax EP decode normally produces uneven `dp_metadata` chunk
  sizes; if yes, the current XPU AG/RS path can hang before any useful speed
  measurement.
- Add focused instrumentation around worker command progress after model load
  and before KV/cache profiling; the next EP blocker is likely worker
  synchronization rather than raw AG/RS collective latency.
- Investigate XPU async sampled-token copy separately from EP. A synchronous
  guarded path may be useful, but it is not sufficient by itself.
- Add focused timing around MiniMax MoE dispatch/all-to-all when EP reaches
  generation.
- If EP can reach generation, tune `E=64,N=1536` for decode `M=1` before any
  throughput comparison.

## EP Attempt 9: Active Package Patch, Async Scheduling On

Earlier patching had been done in `/home/steve/src/vllm`, but the benchmark
venv imports vLLM from:

`/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm`

The active package was patched directly with:

- `VLLM_XPU_PAD_UNEVEN_ALLGATHERV=1` padded XPU `all_gatherv`
- `VLLM_XPU_SYNC_ASYNC_OUTPUT_COPY=1` diagnostic synchronous sampled-token copy
- `VLLM_XPU_TRACE_WORKER_RPC=1` worker RPC progress logging

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n32-20260514T122909Z.log`
- Without the active sync-copy guard, EP again reached sampling and failed in
  `WorkerAsyncOutputCopy` / `update_async_output_token_ids` with:
  `UR_RESULT_ERROR_DEVICE_LOST`.
- This confirmed the prior source-tree patch was not active for the benchmark
  process.
- No benchmark datapoint accepted.

## EP Attempt 10: Active Package Patch, XCCL Group Diagnostics

Added:

- `benchmarks/b70_xccl_group_init_probe.py`

Purpose: determine whether repeated XCCL and Gloo process-group creation is
itself enough to reproduce the low-VRAM EP initialization stalls.

Outcomes:

- Logs:
  - `/home/steve/bench-results/xccl-group-probes/b70-xccl-group-init-probe-ofi-p2p-20260514T124150Z.log`
  - `/home/steve/bench-results/xccl-group-probes/b70-xccl-group-init-probe-ofi-p2p-fabriccheck0-20260514T124300Z.log`
- Eight repeated XCCL groups plus Gloo groups completed with
  `CCL_TOPO_P2P_ACCESS=1`.
- Repeating with `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` also completed.
- Passing `device_id=torch.device("xpu:N")` to `init_process_group` suppresses
  the XCCL rank/device warning, but it can break later Gloo `new_group` calls
  with `No backend type associated with device type xpu`; do not apply that as
  a broad vLLM patch yet.

Interpretation:

- Repeated process-group construction alone is not the EP blocker.
- The low-VRAM EP stalls likely need the fuller vLLM mix of worker RPC,
  group setup, shared-memory broadcast, and model initialization to reproduce.

## EP Attempt 11: Active Package Patch, Fabric Vertex Check Disabled

Command shape added:

```bash
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 \
VLLM_XPU_PAD_UNEVEN_ALLGATHERV=1 \
VLLM_XPU_SYNC_ASYNC_OUTPUT_COPY=1 \
VLLM_XPU_TRACE_WORKER_RPC=1
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n32-20260514T124411Z.log`
- This was the furthest EP run: model load, profiling, KV init, warmup, prompt
  rendering, and first decode `sample_tokens` completed.
- It stalled in the next `sample_tokens`.
- GDB showed rank 0 blocked inside the diagnostic synchronous XPU-to-CPU copy:
  `urEnqueueUSMMemcpy` -> `at::native::xpu::_copy_xpu` -> Python `.to("cpu")`.
- The sync-copy patch moves the async `DEVICE_LOST` failure into a Level Zero
  memcpy wait; it is a diagnostic, not a usable fix.
- No benchmark datapoint accepted.

## EP Attempt 12: Active Package Patch, Async Scheduling Disabled

Command shape used the same topology/patch environment as Attempt 11 and added:

```bash
--no-async-scheduling
```

Outcome:

- Log:
  `/home/steve/bench-results/minimax-m2.7-ep-exploration/vllm-minimax-m27-autoround-tp4-p512n32-20260514T124752Z.log`
- The run stalled early at low VRAM after rank assignment, before normal model
  load.
- No benchmark datapoint accepted.

## Updated EP Read

The padded XPU allgatherv path still looks like a real correctness fix for a
synthetic XCCL hang, and `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` can help
one EP path get further. The full MiniMax EP path is still blocked before a
quality-valid result:

- Async scheduling on reaches decode, but sampled-token CPU transfer can
  `DEVICE_LOST` or hang in Level Zero.
- Async scheduling off avoids that copy path but stalls before model load on
  current runs.
- Worker/group construction microbenchmarks pass, so the failure is likely an
  interaction in vLLM's EP worker execution rather than a simple process-group
  creation failure.

EP remains a high-upside workstream, but it is not the next promotable path.
The immediate quality-preserving performance path is to repair the faster TP4
compiled/AOT recipe, which previously reached about `73` output tok/s but was
invalidated by semantic quality corruption.
