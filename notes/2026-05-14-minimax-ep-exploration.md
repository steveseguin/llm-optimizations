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

## Current EP Read

EP is still worth pursuing, but it is not a drop-in speed win on this stack yet.
The current blockers are:

- EP/all-to-all initialization is unreliable on the current XPU+oneCCL path.
- The EP-local MoE shape needs a real tuned XPU config and probably kernel
  validation, not just copied TP-only parameters.
- The available all-to-all backend used here,
  `allgather_reducescatter`, may be too expensive or unstable on four PCIe B70s.

Do not submit these EP attempts to LocalMaxxing; they are failure/diagnostic
results only.

## Next EP Work

- Try `--enforce-eager` with EP and no XPU graph to separate all-to-all setup
  from graph capture.
- Try `--all2all-backend naive` as a correctness/control path if vLLM accepts it
  on XPU.
- Add focused timing around MiniMax MoE dispatch/all-to-all when EP reaches
  generation.
- If EP can reach generation, tune `E=64,N=1536` for decode `M=1` before any
  throughput comparison.
