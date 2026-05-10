# MiniMax M2.7 DP4+EP CCL Local-Rank Screen, 2026-05-10

## Context

Goal: test whether MiniMax M2.7 AutoRound INT4 can move from TP4 replication to
DP4+expert-parallel weight filtering on four Arc Pro B70 32GB cards. The
quality-preserving attraction is obvious: if each card owns only 64 of 256
experts, memory and MoE work should fall enough to justify higher targets than
the current TP4 reference.

The near-term aspiration for AutoRound is now `60 tok/s` p512/n1536 output
without model-quality changes. Verified speculative/MTP paths should target
`75+ tok/s` and include acceptance/verification details before being promoted.

## Patch Under Test

Two local vLLM changes were active:

- XPU worker now sets oneCCL local-rank environment from DP/TP/PP topology before
  distributed initialization.
- The llm-scaler U4 MoE EP path optionally maps non-local expert ids to zero
  after zeroing their weights. This is diagnostic only; the llm-scaler tiny
  kernels already check `expert < 0`, so this may slow EP and should not be
  treated as permanent without larger validation.

Patch record:

- `patches/vllm-xpu-dp4-ep-ccl-localrank-and-moe-safeids-20260510.patch`

## Results

### DP4+EP no-scaler smoke

Command shape:

```bash
TP=1 DTYPE=float16 MAX_MODEL_LEN=64 MAX_BATCHED_TOKENS=64 MAX_NUM_SEQS=1 \
CCL_ATL_TRANSPORT=ofi CCL_PROCESS_LAUNCHER=none CCL_P2P=0 \
USE_LLM_SCALER_MOE=0 \
EXTRA_SERVER_ARGS="--data-parallel-size 4 --data-parallel-size-local 4 --enable-expert-parallel --enable-ep-weight-filter --all2all-backend allgather_reducescatter --api-server-count 1 --enforce-eager" \
INPUT_LEN=16 OUTPUT_LEN=8 NUM_PROMPTS=1 \
scripts/bench-vllm-minimax-autoround-serve-xpu.sh
```

Outcome:

- Completed one request.
- Output throughput: `4.321792 tok/s`.
- Total throughput: `12.965376 tok/s`.
- Mean TTFT: `936.326 ms`.
- Mean TPOT: `130.626 ms`.

Interpretation: DP4+EP now initializes and can generate with the CCL local-rank
patch. This is a stability result, not a performance result.

### DP4+EP scaler safe-id smoke

Same command shape with `USE_LLM_SCALER_MOE=1`.

Outcome:

- Completed one request.
- Output throughput: `2.995137 tok/s`.
- Total throughput: `8.985410 tok/s`.
- Mean TTFT: `1146.684 ms`.
- Mean TPOT: `217.697 ms`.

Interpretation: the safe-id path is slower in the tiny eager smoke. Since the
kernel already handles negative expert ids, the safe-id patch should remain a
diagnostic branch unless a larger run proves it avoids a real correctness issue.

### DP4+EP compiled mode

Tried p64/n128 compiled serving with `MAX_MODEL_LEN=256`,
`MAX_BATCHED_TOKENS=128`, `gpu_memory_utilization=0.98`, max autotune disabled,
coordinate-descent tuning disabled, CCL P2P disabled.

Outcome:

- Failed during Inductor autotuning/compile memory probe.
- Error: attempted to allocate about `1.15 GiB` with only about `677 MiB` free.
- Log reported about `30.81 GiB` already allocated by PyTorch after model load.

Interpretation: compiled DP4+EP is memory blocked on 32GB B70s in the current
runtime shape. The model fits per-rank only barely, leaving too little room for
compile scratch and KV cache. Fast NVMe does not address this bottleneck.

## Decision

Keep the CCL local-rank patch in the reproduction archive because it is a real
XPU DP/EP initialization fix. Do not submit these EP smoke results to
LocalMaxxing; they are useful engineering data but not a meaningful leaderboard
result.

For performance, the next best path is back to TP4 quality-preserving source
work:

- Q/K RMS variance allreduce boundary fusion.
- Hidden-state allreduce plus residual/RMSNorm fusion.
- Projection and MoE epilogue fusion around the collective wait sites.
- Lower-overhead timing around attention/KV and MoE to choose the first boundary
  worth porting to C++/SYCL.

