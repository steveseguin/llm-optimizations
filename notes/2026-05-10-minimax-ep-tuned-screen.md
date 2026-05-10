# MiniMax M2.7 AutoRound EP Screen, 2026-05-10

## Purpose

Screen vLLM's built-in expert parallelism for MiniMax M2.7 on the four B70
system after the standard TP4 path had reached a `41.130667` output tok/s
fast AOT result at p512/n1536. That fast AOT result is now suspect because its
cached graph did not visibly include Q/K RMS variance allreduce; the current
quality-conservative TP4 reference is `37.552538` output tok/s. vLLM's MiniMax recipe recommends TP4+EP4 on H100-class
systems, and the EP deployment docs describe expert layers as sharded across
EP ranks while attention remains tensor-parallel when TP is greater than one.

References:

- https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
- https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/

## Setup

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Engine: vLLM `0.20.1-local`, XPU/Level Zero
- Hardware: 4x Intel Arc Pro B70 32GB
- Parallelism: `--tensor-parallel-size 4 --enable-expert-parallel`
- Quantization: INT4 AutoRound W4A16 through the local INC/MoeWNA16 patch
- MoE decode: llm-scaler unsigned INT4 decode path enabled
- Attention: FlashAttention2, auto KV dtype
- No speculative decoding, no expert dropping, no power-limit changes

The important EP config file is:

`/home/steve/bench-results/minimax-m2.7-autoround-vllm/moe-config-ep-hybrid-m1-default-prefill/E=64,N=1536,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json`

## Results

| Run | Prompt | Output | KV tokens | Output tok/s | Total tok/s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| EP smoke, default MoE config | 1 | 8 | 8,896 | 3.845 | 4.326 | Functional only |
| EP p512/n512, default MoE config | 512 | 512 | 8,768 | 25.076 | 50.152 | Negative |
| EP p512/n512, tuned E64/N1536 config | 512 | 512 | 16,320 | 29.892 | 59.785 | Improved, still below TP4 |
| EP p512/n1536, tuned E64/N1536 config | 512 | 1536 | 16,320 | 30.911 | 41.214 | Submitted diagnostic |
| EP round-robin requested | 512 | 512 | 8,768 | 24.998 | 49.995 | vLLM fell back to linear |
| EP + DBO requested | 512 | 512 | n/a | n/a | n/a | Blocked before load |

LocalMaxxing accepted the tuned p512/n1536 diagnostic result as
`cmozofyv5005hlo01puv9rjs6`. The first POST failed because LocalMaxxing rejects
`backend=xpu`; the accepted payload omits `backend` and records XPU details in
notes and `engineFlags.extraFlags`.

## Interpretation

The tuned EP MoE config matters. Without it, vLLM uses the default MoE config
for `E=64,N=1536`, KV headroom drops to about 8.8k tokens, and p512/n512 is only
about 25 output tok/s. Pointing `VLLM_TUNED_CONFIG_FOLDER` at the E64/N1536 seed
recovers 16.3k KV tokens and improves p512/n512 to about 29.9 output tok/s.

That still does not beat the non-EP TP4 path. The best EP run here, p512/n1536,
is 30.911 output tok/s versus the Q/K-allreduce quality-conservative TP4
reference of 37.553 output tok/s. On B70/XPU, EP's AgRs all-to-all cost and
scheduling overhead outweigh the reduced per-rank expert count for batch-1
single-session decode.

Round-robin placement is not useful in the current vLLM MiniMax topology. The
runtime warns that round-robin expert placement is only supported for models
with multiple expert groups and no redundant experts, then falls back to linear
placement.

Dual Batch Overlap is also blocked on this XPU path. vLLM asserts that
microbatching currently requires `deepep_low_latency` or
`deepep_high_throughput`, while XPU uses the AgRs/allgather-reducescatter
manager.

## Next Work

Keep EP as a documented diagnostic path, not the recommended recipe. Future EP
work would need an XPU-specific low-latency all-to-all or a model-specific
active-expert path that avoids generic AgRs transfer overhead. The more likely
quality-preserving speed path remains source-level fusion around MiniMax Q/K
allreduce+RMS, RoPE/KV writes, and compiled graph boundaries.
