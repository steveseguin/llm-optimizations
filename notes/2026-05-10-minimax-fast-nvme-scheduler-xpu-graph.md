# MiniMax M2.7: Fast NVMe, Scheduler Shape, and XPU Graph

## Context

After reboot, a Samsung 9100 PRO 1TB NVMe was formatted as ext4 and mounted at `/mnt/fast-ai`. The MiniMax M2.7 AutoRound checkpoint was copied from `/mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround` to `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`.

The copy moved 120,759,303,359 bytes at about 398 MB/s. The checkpoint reports as 112.43 GiB in vLLM.

## Main Results

All results use:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Engine: vLLM `0.20.1-local`, XPU/Level Zero
- Hardware: 4x Intel Arc Pro B70 32GB, AMD EPYC 9015, 16GB RAM
- Precision: FP16 activations, AutoRound W4A16 weights
- Parallelism: TP4, `distributed-executor-backend mp`
- Patch path: `VLLM_XPU_USE_LLM_SCALER_MOE=1` with llm-scaler u4 decode-only MoE bridge
- Power: stock limits, no power-limit increase

| Run | Max Batched Tokens | Extra | KV Tokens | Output tok/s | Total tok/s | Load |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| p512/n1024 best | 1024 | default | 17,216 | 36.670 | 55.005 | 84.39s |
| p512/n1024 old USB baseline | 1024 | default | 17,216 | 35.933 | 53.900 | 348.18s |
| p512/n1024 | 1536 | negative | 9,216 | 29.478 | 44.217 | 86.79s |
| p512/n1024 | 512 | negative | 9,472 | 31.862 | 47.793 | 84.04s |
| p512/n1024 | 1024 | `--gpu-memory-utilization 0.95` | 33,408 | 36.017 | 54.025 | 80.93s |
| p512/n512 | 1024 | default | 17,216 | 35.297 | 70.594 | 74.95s |
| p512/n512 | 1024 | `XPU_GRAPH=1`, negative | 9,408 | 26.316 | 52.632 | 78.06s |
| p1/n1024 | 1024 | default | 9,408 | 31.880 | 31.911 | 77.13s |

## Findings

The NVMe move is a large iteration-speed win. Comparable p512/n1024 load time fell from 348.18 seconds on the external NTFS drive to 84.39 seconds on ext4 NVMe. Decode speed also slightly improved on the comparable shape: 36.67 output tok/s versus 35.93.

`--max-num-batched-tokens` is not a harmless benchmark detail. The original good p512/n1024 run used 1024. Re-running with 1536 looked like a post-reboot regression, but the change reduced KV cache from 17,216 to 9,216 tokens and dropped output from 36.67 to 29.48 tok/s. Setting it lower at 512 was also negative, at 31.86 output tok/s.

`XPU_GRAPH=1` remains negative for this TP4 path. vLLM reports that XPU graph does not support communication-op capture and disables graph mode. The run still lost KV headroom and p512/n512 fell from 35.30 to 26.32 output tok/s.

`--gpu-memory-utilization 0.95` is useful for context capacity rather than raw speed. It increased KV cache from 17,216 to 33,408 tokens and still held 36.02 output tok/s on p512/n1024. For raw decode speed, default gpu-memory-utilization is slightly better.

PCIe reporting is still odd at the endpoint level. The B70 endpoints and downstream internal Intel bridge ports report 2.5 GT/s x1, but the root-to-card upstream links report PCIe 5.0 x16. Treat the endpoint x1 field as an internal/reporting artifact unless a direct bandwidth test proves otherwise.

## Current Recommendation

For fast single-session MiniMax M2.7 testing:

```bash
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm \
HF_HOME=/mnt/fast-ai/llm-cache/hf \
USE_LLM_SCALER_MOE=1 \
CCL_IPC=default \
XPU_GRAPH=0 \
DTYPE=float16 \
INPUT_LEN=512 \
OUTPUT_LEN=1024 \
MAX_MODEL_LEN=2048 \
MAX_BATCHED_TOKENS=1024 \
MAX_NUM_SEQS=1 \
NUM_PROMPTS=1 \
TP=4 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

For larger-context serving where KV capacity matters more than peak decode:

```bash
GPU_MEMORY_UTILIZATION=0.95 \
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm \
HF_HOME=/mnt/fast-ai/llm-cache/hf \
USE_LLM_SCALER_MOE=1 \
CCL_IPC=default \
XPU_GRAPH=0 \
DTYPE=float16 \
INPUT_LEN=512 \
OUTPUT_LEN=1024 \
MAX_MODEL_LEN=2048 \
MAX_BATCHED_TOKENS=1024 \
MAX_NUM_SEQS=1 \
NUM_PROMPTS=1 \
TP=4 \
/home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

## Next Work

- Test whether `gpu-memory-utilization` around 0.92-0.94 gives most of the KV increase without the small speed loss seen at 0.95.
- Add or tune a MiniMax-specific `fused_moe` config for `E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16`.
- Keep XPU graph disabled for TP4 until communication-op capture support changes.
- Prefer `/mnt/fast-ai` for active benchmarks and use the external drive as colder storage.
