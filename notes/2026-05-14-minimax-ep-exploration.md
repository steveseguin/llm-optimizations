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

## Next EP Work

- Stop treating `naive` as available unless the current vLLM all-to-all registry
  is patched or an older backend is restored.
- Inspect vLLM's EP all-to-all backend registration and XPU/xccl interaction
  before launching more full MiniMax EP probes.
- Add a smaller synthetic all-to-all/XCCL repro for the `E=64,N=1536` path so
  the communication layer can be debugged without 400+ GiB of model load churn.
- Add focused timing around MiniMax MoE dispatch/all-to-all when EP reaches
  generation.
- If EP can reach generation, tune `E=64,N=1536` for decode `M=1` before any
  throughput comparison.
