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
| p512/n1024 | 1024 | installed B70 MoE config, run 1 | 17,216 | 36.876 | 55.314 | 73.17s |
| p512/n1024 | 1024 | installed B70 MoE config, run 2 | 17,216 | 36.620 | 54.930 | 78.06s |
| p512/n1024 | 1024 | `--gpu-memory-utilization 0.93`, unstable | 22,592 | hung | hung | 65.82s |
| p512/n1024 | 1024 | BF16 default memory, failed | 0 | no KV cache | no KV cache | 66.09s |
| p512/n1024 | 1024 | BF16, `--gpu-memory-utilization 0.95`, run 1 | 18,880 | 37.304 | 55.955 | 77.30s |
| p512/n1024 | 1024 | BF16, `--gpu-memory-utilization 0.95`, run 2 | 18,880 | 35.954 | 53.931 | 67.17s |

## Findings

The NVMe move is a large iteration-speed win. Comparable p512/n1024 load time fell from 348.18 seconds on the external NTFS drive to 84.39 seconds on ext4 NVMe. Decode speed also slightly improved on the comparable shape: 36.67 output tok/s versus 35.93.

`--max-num-batched-tokens` is not a harmless benchmark detail. The original good p512/n1024 run used 1024. Re-running with 1536 looked like a post-reboot regression, but the change reduced KV cache from 17,216 to 9,216 tokens and dropped output from 36.67 to 29.48 tok/s. Setting it lower at 512 was also negative, at 31.86 output tok/s.

`XPU_GRAPH=1` remains negative for this TP4 path. vLLM reports that XPU graph does not support communication-op capture and disables graph mode. The run still lost KV headroom and p512/n512 fell from 35.30 to 26.32 output tok/s.

`--gpu-memory-utilization 0.95` is useful for context capacity rather than raw speed. It increased KV cache from 17,216 to 33,408 tokens and still held 36.02 output tok/s on p512/n1024. For raw decode speed, default gpu-memory-utilization is slightly better.

`--gpu-memory-utilization 0.93` is currently unstable. It increased KV cache to 22,592 tokens, but generation did not advance and vLLM emitted two shared-memory broadcast wait warnings before manual interruption. Do not use 0.93 as a serving setting without a repeat/profiling pass.

BF16 is viable only with a higher memory target on this setup. The default-memory BF16 run loaded the model but reported negative KV headroom and failed before generation with `No available memory for the cache blocks`. With `--gpu-memory-utilization 0.95`, BF16 completed twice at 37.304 and 35.954 output tok/s, with 18,880 KV tokens. This is a quality-conservative capacity mode rather than a clear speed breakthrough: it keeps BF16 activations and has enough context headroom, but the two-run range overlaps the FP16 best path.

LocalMaxxing accepted the first BF16 0.95 run as `cmoz632kr0068tl017a1z6r0u`. The submission notes include the repeat at 35.954 output tok/s so the public record does not hide the observed variance.

PCIe reporting is still odd at the endpoint level. The B70 endpoints and downstream internal Intel bridge ports report 2.5 GT/s x1, but the root-to-card upstream links report PCIe 5.0 x16. Treat the endpoint x1 field as an internal/reporting artifact unless a direct bandwidth test proves otherwise.

The archived hybrid B70 MoE config was not installed under vLLM's exact expected filename after reboot. Installing it to:

```text
/home/steve/.venvs/vllm-xpu/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json
```

removed the missing-config warning and produced two p512/n1024 repeats at 36.876 and 36.620 output tok/s. That is neutral-to-small-positive against the 36.670 no-config result, so keep the config installed for warning cleanup and possible prefill stability, but do not treat it as a major optimization. Reinstall with:

```bash
/home/steve/llm-optimizations-publish/scripts/install-minimax-b70-moe-config.sh
```

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

For larger-context serving where KV capacity and a more quality-conservative activation dtype matter more than peak decode:

```bash
GPU_MEMORY_UTILIZATION=0.95 \
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm \
HF_HOME=/mnt/fast-ai/llm-cache/hf \
USE_LLM_SCALER_MOE=1 \
CCL_IPC=default \
XPU_GRAPH=0 \
DTYPE=bfloat16 \
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

- Treat `gpu-memory-utilization` default as the raw-speed setting and 0.95 as the larger-KV setting; 0.93 hung and needs lower-level profiling before more midpoint sweeps.
- Keep BF16 0.95 as the quality-conservative MiniMax serving recipe; the BF16 default-memory path has no usable KV cache headroom on this host.
- Add or tune a MiniMax-specific `fused_moe` config for `E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16`.
- Keep XPU graph disabled for TP4 until communication-op capture support changes.
- Prefer `/mnt/fast-ai` for active benchmarks and use the external drive as colder storage.
