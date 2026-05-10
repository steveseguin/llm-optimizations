# MiniMax M2.7: Q/K Allreduce Diagnostic and 41 tok/s Repeat

Date: 2026-05-10

Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`

Runtime: vLLM `0.20.1-local`, XPU/Level Zero, TP4 on 4x Intel Arc Pro B70, FP16 activations, `USE_LLM_SCALER_MOE=1`, installed B70 MoE config, `MAX_MODEL_LEN=2048`, `MAX_BATCHED_TOKENS=1024`, `MAX_NUM_SEQS=1`, XPU graph disabled, stock power limits.

## Results

| Run | Env | Prompt | Output | KV tokens | Output tok/s | Total tok/s | Log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q/K allreduce skip diagnostic | `VLLM_MINIMAX_QK_SKIP_TP_ALLREDUCE=1` | 512 | 256 | 13,632 | 26.158 | 78.474 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T031650Z.log` |
| Control after recompile | normal Q/K TP allreduce | 512 | 256 | 17,216 | 37.549 | 112.647 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n256-20260510T031944Z.log` |
| Control p512/n512 run 1 | normal Q/K TP allreduce | 512 | 512 | 17,216 | 39.611 | 79.221 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T032211Z.log` |
| Control p512/n512 run 2 | normal Q/K TP allreduce | 512 | 512 | 17,216 | 39.516 | 79.033 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n512-20260510T032448Z.log` |
| Control p512/n1024 run 1 | normal Q/K TP allreduce | 512 | 1024 | 17,216 | 39.894 | 59.841 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1024-20260510T033148Z.log` |
| Control p512/n1024 run 2 | normal Q/K TP allreduce | 512 | 1024 | 17,216 | 40.304 | 60.456 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1024-20260510T033429Z.log` |
| Control p512/n1536 run 1 | normal Q/K TP allreduce | 512 | 1536 | 17,216 | 40.864 | 54.486 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T033913Z.log` |
| Control p512/n1536 run 2 | normal Q/K TP allreduce | 512 | 1536 | 17,216 | 41.131 | 54.841 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T034207Z.log` |
| `max_model_len=4096` capacity | normal Q/K TP allreduce | 512 | 1536 | 9,408 | 33.258 | 44.344 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T034709Z.log` |

LocalMaxxing accepted the first p512/n512 control run as `cmoz7rs2w0077tl01o3f1kxnm`, the second p512/n1024 control run as `cmoz82i2f007itl01fkno9or1`, the second p512/n1536 control run as `cmoz8cow60001pd010klrb8g8`, and the 4096-context capacity run as `cmoz8k9z40008pd01rhu50c0n`.

## Interpretation

The Q/K TP-allreduce skip is intentionally correctness-breaking. It was only meant to bound the upside from removing the tiny Q/K variance collective. It was much slower than the normal path and also reduced KV headroom, so it is not a useful approximation of the CUDA Lamport fused op and should stay diagnostic-only.

The valid p512/n512 control repeated tightly at `39.61` then `39.52` output tok/s. The longer p512/n1024 shape repeated at `39.89` then `40.30` output tok/s, and the full 2048-token request window p512/n1536 repeated at `40.86` then `41.13` output tok/s. These use the normal Q/K TP allreduce, normal MiniMax math, the same AutoRound W4A16 target weights, no speculative decode, and no power-limit change. The likely contributors versus older p512/n512 and p512/n1024 records are the fast NVMe setup, installed B70 MoE config, current warmed compile/cache state, and the stable `max-num-batched-tokens=1024` scheduler shape.

Raising `max_model_len` to 4096 is a capacity tradeoff, not a speed path. The same p512/n1536 request fell to `33.26` output tok/s because the available GPU KV cache dropped from 17,216 to 9,408 tokens and max concurrency fell to 2.30x for 4096-token requests.

## Next Work

- Do not pursue Q/K allreduce elimination by dropping the collective; it is invalid and slower.
- A real XPU equivalent of CUDA `minimax_allreduce_rms_qk` would still need to fuse local variance, inter-rank exchange, and RMS apply in one graph-safe custom op. The skip diagnostic shows that a naive removal is not a useful proxy.
- Next test whether `gpu_memory_utilization=0.95` can recover some 4096-context KV headroom without falling back to the older mid-30 tok/s range.
