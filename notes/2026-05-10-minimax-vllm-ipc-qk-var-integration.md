# MiniMax vLLM XPU IPC Q/K Variance Integration

## Summary

I added a default-off vLLM hook for MiniMax M2.7 that replaces the tiny
`[tokens, 2]` Q/K variance TP allreduce with the Level Zero IPC mailbox
prototype. The hook lives in the normal `MiniMaxText01RMSNormTP.forward_qk`
path, so it does not require the slower standalone Q/K var/apply helper.

Enable it with:

```bash
export VLLM_MINIMAX_QK_RMS_XPU_IPC=1
export VLLM_MINIMAX_QK_RMS_XPU_IPC_MAX_TOKENS=1024
export VLLM_MINIMAX_QK_RMS_XPU_IPC_SLOTS=128
export VLLM_MINIMAX_QK_RMS_XPU_IPC_TIMEOUT_ITERS=100000
```

Patch artifact:

- `patches/vllm-minimax-xpu-ipc-qk-var-20260510.patch`

## Result

Eager smoke:

```bash
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ipc-smoke \
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
HF_HOME=/mnt/fast-ai/llm-cache/hf USE_LLM_SCALER_MOE=1 DTYPE=float16 \
INPUT_LEN=1 OUTPUT_LEN=4 MAX_MODEL_LEN=128 MAX_BATCHED_TOKENS=64 \
MAX_NUM_SEQS=1 NUM_PROMPTS=1 TP=4 VLLM_MINIMAX_QK_RMS_XPU_IPC=1 \
EXTRA_ARGS='--enforce-eager' \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Log evidence:

```text
MiniMax Q/K XPU IPC allreduce initialized (world_size=4, slots=128, max_tokens=1024)
Throughput: 1.52 requests/s, 7.59 total tokens/s, 6.07 output tokens/s
```

This is a liveness/correctness smoke only. Eager mode is far slower than the
compiled MiniMax speed path and should not be submitted to LocalMaxxing.

## Negative Compile Attempt

Allowing the IPC path during torch.compile failed during engine initialization:

```text
Skip inlining `torch.compiler.disable()`d function
Developer debug context: _MiniMaxQkRmsXpuIpcState.ensure_initialized
```

I tightened the guard so compiled runs fall back to the default oneCCL
`tensor_model_parallel_all_reduce` instead of crashing. A follow-up compiled
smoke with the IPC flag set completed, but the IPC initialization log did not
appear, confirming fallback.

## Next Work

The graph-safe version needs IPC setup outside the compiled forward path:

- allocate mailboxes and exchange Level Zero handles during worker/model setup;
- keep only fixed pointer tensors, device counters, and the custom op call in
  the compiled region;
- then rerun p1/n8 logits comparison against the default oneCCL path;
- only after logits match, benchmark p512/n512 and p512/n1536.

Until that exists, keep `VLLM_MINIMAX_QK_RMS_XPU_IPC` unset for real MiniMax
throughput runs.
