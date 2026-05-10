# MiniMax M2.7 Timing and oneCCL Sweep, 2026-05-10

## Scope

These runs followed the fast-NVMe MiniMax M2.7 AutoRound INT4 vLLM/XPU TP4 result. The objective was to find the next decode bottleneck after the unsigned llm-scaler u4 MoE bridge moved p512/n1536 to 41.130667 output tok/s.

Model:

- `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70
- `USE_LLM_SCALER_MOE=1`
- `XPU_GRAPH=0`
- `DTYPE=float16`
- no speculative decode
- no power-limit changes

## Timing Diagnostics

Synchronized timing is intentionally not a benchmark. It adds device syncs and changes throughput, but it is useful for ranking remaining work.

Compiled p512/n64 timing summary:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n64-20260510T041444Z.log`
- Total throughput under timing: 158.58 total tok/s, roughly 17.62 output tok/s.
- `runner.forward`: 64 calls, avg 45.675 ms.
- `moe.quant_apply`: 1564 ms total.
- `moe.fused_experts_fallback`: 615.9 ms total.
- `moe.llm_scaler_u4_bridge`: 601.6 ms total.
- `moe.router_select`: 270.5 ms total.

Eager p512/n4 per-label timing:

- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n4-20260510T041943Z.log`
- The last steady rank-0 samples around layer count 372 showed:
  - `minimax.attn.qk_norm`: 0.464702 ms/layer.
  - `tp.all_reduce.direct`: usually 0.084-0.088 ms/call, with occasional higher samples.
  - `minimax.attn.kv_attention`: 0.285331 ms/layer.
  - `attn.unified_attention_total`: 0.091680 ms/layer.
  - `moe.llm_scaler_u4_bridge`: 0.105740 ms/layer on decode-token MoE calls.
  - `minimax.moe.experts_total`: 0.579922 ms/layer.
  - `runner.forward`: 138.788797 ms for one synchronized eager decode step.

Interpretation:

- The custom u4 MoE bridge is no longer the only ceiling.
- Q/K RMS plus its TP allreduce, attention/KV, output projection allreduce, MoE output allreduce, and graph/dispatch boundaries are all material.
- A naive correctness-breaking Q/K allreduce skip was already negative, so the next valid direction should preserve global TP RMS math.

## oneCCL Environment Sweep

The sweep used Intel oneCCL environment variables documented in the oneCCL developer guide:

- Main environment variables: <https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-16/main.html>
- Environment variable overview: <https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-16/environment-variables.html>
- Low-precision environment variables: <https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-16/low-precision.html>

Baseline for comparison:

- p512/n512 fast-NVMe FP16 path: 79.22117 total tok/s, 39.610585 output tok/s.
- Log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T011816Z.log`

Results:

| Change | Total tok/s | Output tok/s | Log |
| --- | ---: | ---: | --- |
| `CCL_SYCL_ALLREDUCE_TMP_BUF=1` | 78.815625 | 39.407813 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T042255Z.log` |
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | 78.624438 | 39.312219 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T042517Z.log` |
| `CCL_ALLREDUCE_SMALL_THRESHOLD=0` | 77.370316 | 38.685158 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T042741Z.log` |

Conclusion:

- These oneCCL toggles did not improve the current TP4 MiniMax decode path.
- Keep default oneCCL behavior for published MiniMax TP4 runs unless a later patch changes the collective shapes.
- Do not submit these negative oneCCL screens to LocalMaxxing; they are useful implementation notes, not leaderboard-quality improvements.

## Next Work

The next lower-risk code path is a default-off XPU helper that preserves the Q/K variance allreduce but fuses the post-allreduce RMS scale application with RoPE. This does not remove a collective, but it may reduce pointwise/rotary launch overhead and gives us a correctness-contained stepping stone toward a later Lamport-style fused allreduce/RMS/RoPE kernel.

The higher-risk path remains a true XPU equivalent of vLLM's CUDA `minimax_allreduce_rms_qk` path, with strict logits comparison before any throughput run.
