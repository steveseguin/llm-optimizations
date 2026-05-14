# MiniMax KV Block Size Sweep

Goal: check whether changing vLLM KV block size can improve MiniMax M2.7
AutoRound W4A16 TP4 single-session decode without changing model quality.

Common recipe:

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Runtime: vLLM `0.20.1-local c51df4300`, Intel XPU
- GPUs: 4x Intel Arc Pro B70 32GB
- TP: 4
- Prompt/output shape: `512/1536`
- Context: `2048`
- `max_num_batched_tokens=512`
- `max_num_seqs=1`
- TRITON_ATTN
- Full-decode-only XPU graph
- llm-scaler INT4 MoE decode path enabled
- MiniMax attention delayed allreduce enabled
- No power-limit changes

## Results

| Block size | Output tok/s | Total tok/s | JSON |
| --- | ---: | ---: | --- |
| 128 | 56.2247 | 74.9662 | `/home/steve/bench-results/minimax-m2.7-blocksize-sweep/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T112329Z.json` |
| 256 | 61.0808 mean | 81.4411 mean | `/home/steve/bench-results/minimax-m2.7-quality-gated/minimax-full-decode-graph-triton-tp4-ctx2048-mbt512-bs256-p512n1536-20260514T110046Z-summary.json` |
| 512 | 56.9353 | 75.9137 | `/home/steve/bench-results/minimax-m2.7-blocksize-sweep/vllm-minimax-m27-autoround-tp4-p512n1536-20260514T112613Z.json` |

## Conclusion

Block size `256` remains the best tested value. Both smaller and larger block
sizes cost about `4` to `5` output tok/s versus the quality-gated TP4 baseline.

No LocalMaxxing submissions were made for block sizes `128` or `512` because
they are slower diagnostic runs, not improved or leaderboard-worthy results.
