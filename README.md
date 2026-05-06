# LLM Optimizations Lab

Reproducibility notes, benchmark payloads, and local patches from the Intel Arc Pro B70 Qwen3.6 27B optimization work.

## Current B70 Findings

- Host: Ubuntu 24.04.4 LTS, kernel 6.17.0-23-generic.
- GPUs: 4x Intel Arc Pro B70 / BMG-G31, 32 GB VRAM each.
- Original quality-preserving target remains Qwen3.6 27B `Q4_0` GGUF on llama.cpp. Current best quality-preserving GGUF result is 46.194 tok/s on three B70s at 512 prompt / 512 output, using SYCL tensor split, Q8 activation cache, fused MMVQ2, single-kernel allreduce, fused allreduce+ADD, and `GGML_SYCL_COMM_SYNC_AFTER=2`.
- Current four-card Q4_0 result is 39.204 tok/s with an assist split (`-ts 1/1/1/0.05`), which improves equal 4x but still trails 3x. Equal four-card split remains a negative scaling diagnostic at 34.929 tok/s.
- Best static FP8 result so far: vLLM/XPU, `vrfai/Qwen3.6-27B-FP8`, local XPU patches, 4x B70 TP4, CPU n-gram speculative decode, 49.582 output tok/s at 512 prompt / 512 output. This preserves target-model quality through verified speculative decoding and is ahead of the current Q4_0 TP3 validation.
- Static FP8 TP4 is also the preferred 32k-context Qwen3.6 27B layout: TP4/PP1 at `max_model_len=32768` reaches 42.996 tok/s for 2048 prompt / 256 output and reports 1,133,163 GPU KV-cache tokens. TP2/PP2 fits but is much slower for batch-1 decode at 26.362 tok/s.
- A focused llama.cpp active-device row-split patch now zeros unselected SYCL devices when row-split buffers are created from a selected device subset. The known Q4 tensor-split path still sanity-checks at 45.065 tok/s on a short 3-B70 run, but row split itself remains unsafe: a 4B `SYCL2/SYCL3` smoke hit `UR_RESULT_ERROR_DEVICE_LOST` in the existing SYCL split matmul path.
- FP8 MTP with a hybrid static target plus dynamic block-FP8 `mtp.safetensors` now loads cleanly with an opt-in local vLLM patch, but the corrected MTP path is too slow (`2.36 tok/s` eager smoke, `1.84 tok/s` compiled smoke) and is not a LocalMaxxing result.
- Earlier strongest raw speed result was an INT4 AutoRound model variant, not the Q4_0 GGUF. It improves speed substantially but has quantization quality tradeoffs relative to FP8/BF16.
- 2026-05-05 follow-ups were negative: Q4 small-F32 allreduce regressed, FP8 TP2/PP2 was not competitive for batch-1 speed, the oneCCL topology override regressed, and MiniMax `MUL_MAT_ID` masking only moved the failure to coarse buffer allocation.
- MiniMax M2.7 UD-IQ4_XS four-B70 work is currently blocked by llama.cpp SYCL row-split expert allocation and `GGML_OP_MUL_MAT_ID` split-buffer execution. The `-ncmoe` staircase and follow-up guard test are documented, but no valid MiniMax throughput result exists yet.

## Layout

- `plans/q4_0-gguf-b70-optimization-plan.md`: active quality-preserving GGUF optimization plan.
- `plans/2026-05-05-negative-followups-addendum.md`: latest plan addendum after the PP2, CCL topology, small-F32, and MiniMax guard screens.
- `notes/b70-llm-lab-notes.md`: running investigation log, benchmarks, TODOs, and lessons learned.
- `notes/2026-05-04-qwen36-fp8-b70-fa2.md`: focused writeup for the Qwen3.6 27B static FP8 / vLLM XPU FA2 result on 4x B70.
- `notes/2026-05-04-vllm-xpu-ngram4-fp8-validation.md`: current best static FP8 n-gram speculative validation.
- `notes/2026-05-04-qwen36-q4-eventbarrier.md`: current best Q4_0 three-B70 event-barrier allreduce validation.
- `notes/2026-05-04-minimax-row-split-ncmoe-staircase.md`: MiniMax row-split expert allocation staircase.
- `notes/2026-05-05-negative-followups.md`: negative follow-up screens and backend bugs found after the current best results.
- `notes/2026-05-06-fp8-mtp-block-fp8-clean.md`: clean-load but slow Qwen3.6 FP8 MTP hybrid follow-up.
- `notes/2026-05-06-llm-scaler-source-mining.md`: llm-scaler ESIMD source-mining notes for the next Q4 kernel/fusion work.
- `notes/2026-05-06-q4-graph-pattern-probe.md`: Q4_0 decode graph probe showing same-activation multi-GEMV fusion opportunities.
- `notes/2026-05-06-q4-active-device-row-split.md`: focused active-device row-split patch and row-split safety failure.
- `data/qwen36-fp8-32k-tp4-vs-pp2-20260506.json`: post-reboot Q4 sanity plus FP8 32k-context TP4 vs TP2/PP2 validation.
- `data/q4-active-device-row-split-20260506.json`: structured active-device row-split patch validation and negative row-split smoke.
- `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`: Q4_0 GGUF Vulkan benchmark sweep harness.
- `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`: Q4_0 GGUF SYCL benchmark sweep harness.
- `scripts/bench-qwen36-b70-single-mtp.sh`: single-B70 vLLM INT4 MTP benchmark wrapper.
- `scripts/bench-qwen36-b70-tp2.sh`: dual-B70 vLLM TP2 benchmark wrapper.
- `scripts/bench-vllm-qwen36-fp8.sh`: reusable Qwen3.6 FP8 vLLM latency wrapper with TP/PP/speculative knobs.
- `scripts/submit_localmaxxing_results.py`: LocalMaxxing submission helper. Requires `LMX_API_KEY` in the environment; no API key is stored in this repo.
- `benchmarks/b70_xccl_allreduce_bench.py`: XPU all-reduce/P2P microbenchmark.
- `data/localmaxxing_payloads.json`: sanitized benchmark payloads submitted or queued for LocalMaxxing.
- `data/qwen36-q4-eventbarrier-20260504.json`: structured Q4_0 event-barrier validation data.
- `data/minimax-m27-row-split-ncmoe-staircase-20260504.json`: structured MiniMax staircase failure data.
- `data/2026-05-05-negative-followups.json`: structured negative follow-up screens.
- `patches/llama-b70-openvino-vulkan.patch`: local llama.cpp OpenVINO/Vulkan exploratory patch set.
- `patches/llama-cpp-sycl-allreduce-event-barrier.patch`: incremental event-barrier allreduce marker patch.
- `patches/llama-cpp-sycl-minimax-mulmatid-guard.patch`: diagnostic MiniMax `MUL_MAT_ID` split-buffer guard patch.
- `patches/llama-cpp-active-device-row-split-current-20260506.patch`: focused row-split selected-device to physical-backend split mapping patch.
- `patches/vllm-xpu-mtp-fallback.patch`: vLLM 0.20.1 XPU speculative/MTP fallback patch.
- `patches/vllm-xpu-force-graph-with-comm-experiment.patch`: failed TP2 graph-capture experiment knob retained as a negative result.
- `patches/vllm-xpu-fa2-compressed-tensors-scalar-scales.patch`: vLLM compressed-tensors singleton attention scale fix for Intel XPU FlashAttention2.
- `patches/vllm-xpu-qwen35-gdn-spec-fallback-contiguous-state.patch`: XPU Gated DeltaNet speculative metadata/fallback patch used by the n-gram runs.

## Notes

The strongest quality-preserving paths are now Q4_0 GGUF TP3 and static FP8 TP4 with verified n-gram speculative decoding. The INT4 AutoRound path remains interesting for maximum speed, but it should be treated separately because it changes quantization quality more aggressively.
