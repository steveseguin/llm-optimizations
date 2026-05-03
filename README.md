# LLM Optimizations Lab

Reproducibility notes, benchmark payloads, and local patches from the Intel Arc Pro B70 Qwen3.6 27B optimization work.

## Current B70 Findings

- Host: Ubuntu 24.04 LTS, kernel 6.17.0-14-generic.
- GPUs: 2x Intel Arc Pro B70 / BMG-G31, 32 GB VRAM each.
- Best single-card result so far: vLLM 0.20.1 XPU, Lorbus Qwen3.6 27B INT4 AutoRound, local MTP fallback patch, about 45.2 output tok/s at 500 input / 256 output and about 41.3 output tok/s at 500 input / 512 output.
- Best dual-card result so far: vLLM 0.20.1 XPU TP2 non-MTP, about 49.1 output tok/s at 500 input / 256 output and about 48.3 output tok/s at 500 input / 512 output.
- GGUF status: llama.cpp SYCL single-card is about 24.6 tok/s; patched Vulkan B70 core count on system Mesa is about 22 tok/s; dual GGUF split is not viable yet.

## Layout

- `notes/b70-llm-lab-notes.md`: running investigation log, benchmarks, TODOs, and lessons learned.
- `scripts/bench-qwen36-b70-single-mtp.sh`: single-B70 vLLM INT4 MTP benchmark wrapper.
- `scripts/bench-qwen36-b70-tp2.sh`: dual-B70 vLLM TP2 benchmark wrapper.
- `scripts/submit_localmaxxing_results.py`: LocalMaxxing submission helper. Requires `LMX_API_KEY` in the environment; no API key is stored in this repo.
- `benchmarks/b70_xccl_allreduce_bench.py`: XPU all-reduce/P2P microbenchmark.
- `data/localmaxxing_payloads.json`: sanitized benchmark payloads submitted or queued for LocalMaxxing.
- `patches/llama-b70-openvino-vulkan.patch`: local llama.cpp OpenVINO/Vulkan exploratory patch set.
- `patches/vllm-xpu-mtp-fallback.patch`: vLLM 0.20.1 XPU speculative/MTP fallback patch.
- `patches/vllm-xpu-force-graph-with-comm-experiment.patch`: failed TP2 graph-capture experiment knob retained as a negative result.

## Notes

The strongest score here is from an INT4 model variant, not the Q4_0 GGUF. It improves speed substantially but has quantization quality tradeoffs relative to fp8/fp16. The local MTP fallback is intended to preserve model semantics for accepted speculative tokens, but it needs quality evals before treating it as production-quality inference.