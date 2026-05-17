# MiniMax M2.7 Current Path Timing Probe

Date: 2026-05-17

Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

Hardware: 4x Intel Arc Pro B70 32GB

Engine: vLLM 0.20.1-local XPU

Current promoted strict baseline remains 61.404 output tok/s, 81.872 total tok/s at p512/n1536, TP4, batch 1, ctx 2048. That result has the current full quality gate behind it and LocalMaxxing ID `cmp9xpe3w04pdo4013acdikt7`.

## What Was Measured

Two rank-0 synchronized timing probes were run with `VLLM_XPU_DECODE_TIMING=1`.

Graph mode:

- Artifact directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/current-path-timing-rank0-sync-p512n256-20260517`
- Prompt/output: p512/n256
- Diagnostic total throughput: 151.404 total tok/s
- Caveat: compiled graph mode hides most model internals, so the visible timings are mostly the logits/output tail.

Visible graph-mode rank-0 timing:

- `logits.local_argmax_pair_all_gather`: 224 calls, 61.808 ms total, 0.276 ms avg
- `logits.local_argmax_reduce`: 224 calls, 31.022 ms total, 0.138 ms avg
- `logits.local_argmax_pair_stack`: 224 calls, 25.698 ms total, 0.115 ms avg
- `logits.local_argmax_local_max`: 224 calls, 18.265 ms total, 0.082 ms avg
- `gpu_model_runner.async_output_tolist`: 224 calls, 8.747 ms total, 0.039 ms avg

Eager mode:

- Artifact directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/current-path-eager-timing-rank0-sync-p512n64-20260517`
- Prompt/output: p512/n64
- Diagnostic total throughput: 64.575 total tok/s
- Caveat: eager mode is slower than the promoted graph path, but it exposes the layer buckets that the graph hides.

Top eager-mode rank-0 timing buckets:

- `minimax.moe.experts_total`: 4084 calls, 2155.400 ms total, 0.528 ms avg
- `minimax.attn.qk_norm`: 4084 calls, 1764.466 ms total, 0.432 ms avg
- `minimax.attn.kv_attention`: 4084 calls, 1165.121 ms total, 0.285 ms avg
- `minimax.attn.delayed_residual_allreduce`: 4084 calls, 795.119 ms total, 0.195 ms avg
- `all_reduce:(1, 3072):torch.float16`: 7992 calls, 757.618 ms total, 0.095 ms avg
- `minimax.moe.router_linear`: 4084 calls, 527.053 ms total, 0.129 ms avg
- `all_reduce:minimax_qk_var:(1, 2):torch.float32`: 3960 calls, 392.243 ms total, 0.099 ms avg
- `minimax.attn.qkv`: 4084 calls, 336.940 ms total, 0.083 ms avg
- `minimax.attn.o_proj`: 4084 calls, 330.325 ms total, 0.081 ms avg
- `minimax.attn.rope`: 4084 calls, 239.768 ms total, 0.059 ms avg
- `gpu_model_runner.async_output_tolist`: 56 calls, 2.709 ms total, 0.048 ms avg

## Interpretation

This confirms that the recent callback/list-conversion experiments were targeting a small tail, not the main bottleneck. The output conversion path is only about 0.04-0.05 ms/token on rank 0.

The next quality-preserving optimization work should focus on:

1. MoE expert dispatch and INT4 expert kernel behavior.
2. Q/K RMS variance allreduce plus RMS application.
3. KV attention and RoPE scheduling.
4. Residual allreduce scheduling or fusion.

This is diagnostic evidence only, so it was not submitted to LocalMaxxing.
