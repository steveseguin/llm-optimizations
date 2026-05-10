# MiniMax M2.7: Q/K Allreduce Diagnostic and Suspect 41 tok/s Repeat

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
| `max_model_len=4096`, `gpu_memory_utilization=0.95` | normal Q/K TP allreduce | 512 | 1536 | 33,408 | 36.616 | 48.822 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T035402Z.log` |
| `max_model_len=8192`, `gpu_memory_utilization=0.95` | normal Q/K TP allreduce | 512 | 1536 | 25,600 | 33.308 | 44.411 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T035902Z.log` |
| `max_model_len=8192`, `gpu_memory_utilization=0.95`, larger prompt | normal Q/K TP allreduce | 4096 | 512 | 33,408 | 31.287 | 281.587 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p4096n512-20260510T040609Z.log` |
| `max_model_len=8192`, `gpu_memory_utilization=0.95`, warmed refresh | normal Q/K TP allreduce | 512 | 1536 | 33,408 | 36.805 | 49.074 | `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T040846Z.log` |

LocalMaxxing accepted the first p512/n512 control run as `cmoz7rs2w0077tl01o3f1kxnm`, the second p512/n1024 control run as `cmoz82i2f007itl01fkno9or1`, the second p512/n1536 control run as `cmoz8cow60001pd010klrb8g8`, the 4096-context default-memory capacity run as `cmoz8k9z40008pd01rhu50c0n`, the 4096-context 0.95-memory capacity run as `cmoz8ryb9000bpd014xhl3pxu`, the first 8192-context 0.95-memory capacity run as `cmoz90lg0000wpd018x3zuukw`, the p4096/n512 8192-context run as `cmoz97d350015pd01smqui7lk`, and the warmed p512/n1536 8192-context refresh as `cmoz9ayax001cpd01xkr0w54l`.

Later update: a direct audit of the fast `c15860...` AOT graph moved the p512/n512, p512/n1024, and p512/n1536 fast records from this note to suspect status. The graph contains the regular hidden-state allreduces but does not visibly contain the per-layer Q/K RMS variance allreduce. The corrected quality-conservative p512/n1536 reference is `37.552538` output tok/s and was submitted separately as LocalMaxxing `cmozow03v005wlo01q81bnspx`.

## Interpretation

The Q/K TP-allreduce skip is intentionally correctness-breaking. It was only meant to bound the upside from removing the tiny Q/K variance collective. It was much slower than the normal path and also reduced KV headroom, so it is not a useful approximation of the CUDA Lamport fused op and should stay diagnostic-only.

The p512/n512 control repeated tightly at `39.61` then `39.52` output tok/s. The longer p512/n1024 shape repeated at `39.89` then `40.30` output tok/s, and the full 2048-token request window p512/n1536 repeated at `40.86` then `41.13` output tok/s. These remain useful speed/scheduling clues, but they are no longer quality-cleared because the cached graph audit did not find the Q/K RMS variance allreduce in the fast AOT artifact. The current validated target is the later `37.552538` output tok/s Q/K-allreduce run.

Raising `max_model_len` to 4096 is a capacity tradeoff, not a speed path. The same p512/n1536 request fell to `33.26` output tok/s because the available GPU KV cache dropped from 17,216 to 9,408 tokens and max concurrency fell to 2.30x for 4096-token requests. Setting `gpu_memory_utilization=0.95` improves the 4096-context recipe: KV cache rises to 33,408 tokens, max concurrency rises to 8.16x, and output speed recovers to `36.62` tok/s, but it still trails the 2048-window speed path.

At `max_model_len=8192`, `gpu_memory_utilization=0.95` completed cleanly. The first p512/n1536 run reported 25,600 GPU KV-cache tokens and `33.31` output tok/s; a warmed rerun reported 33,408 KV-cache tokens and improved to `36.81` output tok/s. A real larger-prompt p4096/n512 run reached `31.29` output tok/s and `281.59` total tok/s with the same 33,408-token KV cache. This validates an 8192-context capacity configuration on the current TP4 path, but the scheduler/KV cost still keeps it below the 2048 speed path.

## Next Work

- Do not pursue Q/K allreduce elimination by dropping the collective; it is invalid and slower.
- A real XPU equivalent of CUDA `minimax_allreduce_rms_qk` would still need to fuse local variance, inter-rank exchange, and RMS apply in one graph-safe custom op. The skip diagnostic shows that a naive removal is not a useful proxy.
- Next test a larger real prompt at `max_model_len=8192` only if capacity is the goal; for raw speed keep the 2048-window FP16 recipe.
