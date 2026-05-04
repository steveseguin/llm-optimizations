# LLM Optimizations Lab

Reproducibility notes, benchmark payloads, and local patches from the Intel Arc Pro B70 Qwen3.6 27B optimization work.

## Current B70 Findings

- Host: Ubuntu 24.04.4 LTS, kernel 6.17.0-22-generic.
- GPUs: 4x Intel Arc Pro B70 / BMG-G31, 32 GB VRAM each.
- Original quality-preserving target remains Qwen3.6 27B `Q4_0` GGUF on llama.cpp. Current best quality-preserving GGUF result is 43.605 tok/s on three B70s at 512 prompt / 512 output, using SYCL tensor split, Q8 activation cache, single-kernel allreduce, and the event-barrier allreduce marker patch.
- Best static FP8 result so far: vLLM 0.20.1 XPU, `vrfai/Qwen3.6-27B-FP8`, local singleton-scale FlashAttention2 patch, local XPU speculative metadata patch, 4x B70 TP4, n-gram speculative decode with 4 draft tokens and lookup window 2/5, 46.067 output tok/s at 512 prompt / 512 output. This preserves target-model quality through verified speculative decoding and is ahead of the current Q4_0 TP3 validation.
- Earlier strongest raw speed result was an INT4 AutoRound model variant, not the Q4_0 GGUF. It improves speed substantially but has quantization quality tradeoffs relative to FP8/BF16.
- MiniMax M2.7 UD-IQ4_XS four-B70 work is currently blocked by llama.cpp SYCL row-split expert allocation and `GGML_OP_MUL_MAT_ID` split-buffer execution. The `-ncmoe` staircase is documented, but no valid MiniMax throughput result exists yet.

## Layout

- `plans/q4_0-gguf-b70-optimization-plan.md`: active quality-preserving GGUF optimization plan.
- `notes/b70-llm-lab-notes.md`: running investigation log, benchmarks, TODOs, and lessons learned.
- `notes/2026-05-04-qwen36-fp8-b70-fa2.md`: focused writeup for the Qwen3.6 27B static FP8 / vLLM XPU FA2 result on 4x B70.
- `notes/2026-05-04-vllm-xpu-ngram4-fp8-validation.md`: current best static FP8 n-gram speculative validation.
- `notes/2026-05-04-qwen36-q4-eventbarrier.md`: current best Q4_0 three-B70 event-barrier allreduce validation.
- `notes/2026-05-04-minimax-row-split-ncmoe-staircase.md`: MiniMax row-split expert allocation staircase.
- `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`: Q4_0 GGUF Vulkan benchmark sweep harness.
- `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`: Q4_0 GGUF SYCL benchmark sweep harness.
- `scripts/bench-qwen36-b70-single-mtp.sh`: single-B70 vLLM INT4 MTP benchmark wrapper.
- `scripts/bench-qwen36-b70-tp2.sh`: dual-B70 vLLM TP2 benchmark wrapper.
- `scripts/submit_localmaxxing_results.py`: LocalMaxxing submission helper. Requires `LMX_API_KEY` in the environment; no API key is stored in this repo.
- `benchmarks/b70_xccl_allreduce_bench.py`: XPU all-reduce/P2P microbenchmark.
- `data/localmaxxing_payloads.json`: sanitized benchmark payloads submitted or queued for LocalMaxxing.
- `data/qwen36-q4-eventbarrier-20260504.json`: structured Q4_0 event-barrier validation data.
- `data/minimax-m27-row-split-ncmoe-staircase-20260504.json`: structured MiniMax staircase failure data.
- `patches/llama-b70-openvino-vulkan.patch`: local llama.cpp OpenVINO/Vulkan exploratory patch set.
- `patches/llama-cpp-sycl-allreduce-event-barrier.patch`: incremental event-barrier allreduce marker patch.
- `patches/vllm-xpu-mtp-fallback.patch`: vLLM 0.20.1 XPU speculative/MTP fallback patch.
- `patches/vllm-xpu-force-graph-with-comm-experiment.patch`: failed TP2 graph-capture experiment knob retained as a negative result.
- `patches/vllm-xpu-fa2-compressed-tensors-scalar-scales.patch`: vLLM compressed-tensors singleton attention scale fix for Intel XPU FlashAttention2.
- `patches/vllm-xpu-qwen35-gdn-spec-fallback-contiguous-state.patch`: XPU Gated DeltaNet speculative metadata/fallback patch used by the n-gram runs.

## Notes

The strongest quality-preserving paths are now Q4_0 GGUF TP3 and static FP8 TP4 with verified n-gram speculative decoding. The INT4 AutoRound path remains interesting for maximum speed, but it should be treated separately because it changes quantization quality more aggressively.
