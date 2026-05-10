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

Two additional experimental toggles exist and should stay unset for real
benchmarks:

```bash
export VLLM_MINIMAX_QK_RMS_XPU_IPC_COMPILED=1
export VLLM_MINIMAX_QK_RMS_XPU_IPC_SCALAR_SEQ=1
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

## Compile Behavior

Allowing the IPC path during torch.compile failed during engine initialization:

```text
Skip inlining `torch.compiler.disable()`d function
Developer debug context: _MiniMaxQkRmsXpuIpcState.ensure_initialized
```

I tightened the guard so compiled runs fall back to the default oneCCL
`tensor_model_parallel_all_reduce` instead of crashing. A follow-up compiled
smoke with the IPC flag set completed, but the IPC initialization log did not
appear, confirming fallback.

After moving mailbox setup into `MiniMaxM2Model.load_weights`, compiled runs
can preinitialize the IPC state before graph capture. That avoids the Dynamo
initialization error, but compiled IPC remains unusably slow when explicitly
enabled with `VLLM_MINIMAX_QK_RMS_XPU_IPC_COMPILED=1`: a p1/n4 run compiled for
about 128 seconds and then took hundreds of seconds for four output tokens.

The scalar-sequence mailbox op is also negative in vLLM. It passes the
standalone harness, but MiniMax TP4 p1/n4 measured only about `0.03` output
tok/s in eager mode and about `0.02` output tok/s in compiled opt-in mode.

The default behavior is therefore conservative:

- `VLLM_MINIMAX_QK_RMS_XPU_IPC=1` can still run the eager counter-path liveness
  smoke.
- compiled runs preinitialize only when the env flag is set, then fall back to
  oneCCL unless `VLLM_MINIMAX_QK_RMS_XPU_IPC_COMPILED=1` is explicitly set.
- real throughput benchmarks should keep all IPC env flags unset.

## Next Work

The graph-safe version needs IPC setup outside the compiled forward path:

- allocate mailboxes and exchange Level Zero handles during worker/model setup;
- keep only fixed pointer tensors, device counters, and the custom op call in
  the compiled region;
- move the actual Q/K variance exchange lower than the Python-level custom-op
  hook if the current op-launch/fence placement remains slow;
- then rerun p1/n8 logits comparison against the default oneCCL path;
- only after logits match, benchmark p512/n512 and p512/n1536.

Until that exists, keep `VLLM_MINIMAX_QK_RMS_XPU_IPC` unset for real MiniMax
throughput runs.
