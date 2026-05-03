# Intel Arc Pro B70 LLM Lab Notes

Date: 2026-05-03

## Current Best State

Best single-B70 path:

- Engine: vLLM 0.20.1 XPU.
- Model: `Lorbus/Qwen3.6-27B-int4-AutoRound`.
- Patch: local XPU MTP fallback for Qwen3.6 Gated DeltaNet speculative metadata.
- Shape `input=500`, `output=256`: `5.67 s`, about `45.2 output tok/s`, `133.44 total tok/s`.
- Shape `input=500`, `output=512`: `12.40 s`, about `41.3 output tok/s`, `81.60 total tok/s`.

Best dual-B70 path:

- Engine: vLLM 0.20.1 XPU, tensor parallel size 2, non-MTP.
- Shape `input=500`, `output=256`: `5.22 s`, about `49.1 output tok/s`, `144.88 total tok/s`.
- Shape `input=500`, `output=512`: `10.59 s`, about `48.3 output tok/s`, `95.56 total tok/s`.
- TP2 is useful but still well below the desired 80% single-session uplift over the best single-card MTP path.

GGUF paths so far:

- llama.cpp SYCL single-card Qwen3.6 27B Q4_0 GGUF: about `24.6 tok/s` decode.
- llama.cpp Vulkan with B70 core-count patch on system Mesa: about `22 tok/s` decode.
- llama.cpp dual split is not viable locally yet.

## Hardware

- Host: Ubuntu 24.04 LTS, kernel 6.17.0-14-generic.
- CPU: AMD EPYC 9015.
- RAM reported for benchmark payloads: 16 GB.
- GPUs: 2x Intel Arc Pro B70 / BMG-G31, 32 GB each, exposed as `/dev/dri/renderD128` and `/dev/dri/renderD129`.
- Intel compute-runtime release installed: 26.14.37833.4.
- IGC: 2.32.7.
- Level Zero loader/tools: installed and visible to OpenVINO/SYCL.
- oneAPI compiler/MKL/DNNL 2026.0 installed.

## XCCL P2P Microbenchmark

With `CCL_TOPO_P2P_ACCESS=1`:

- 4 KiB: 0.016 ms.
- 64 KiB: 0.014 ms.
- 1 MiB: 0.029 ms, about 35.9 GB/s.
- 16 MiB: 0.408 ms, about 41.2 GB/s.
- 64 MiB: 1.615 ms, about 41.6 GB/s.
- 256 MiB: 6.417 ms, about 41.8 GB/s.

With P2P disabled, large all-reduce falls to about 5 GB/s. Communication bandwidth is not the main TP2 blocker; vLLM's disabled XPU graph capture around communication is the larger issue.

## vLLM Patch Notes

`patches/vllm-xpu-mtp-fallback.patch` changes vLLM 0.20.1 so the XPU Gated DeltaNet path falls back to the generic `_forward_core` path when speculative sequence masks are present. This avoids the original XPU speculative assert and makes MTP usable on a single B70.

`patches/vllm-xpu-force-graph-with-comm-experiment.patch` is a negative result. It adds a guarded environment override for XPU graph capture with TP communication, but forcing it hits a CUDA-specific communicator assertion in vLLM internals. The useful follow-up is not this flag; it is designing an XPU-safe graph/communication capture boundary or XPU communicator equivalent.

## llama.cpp / OpenVINO Findings

OpenVINO 2026.1 L0 source build worked after adding a missing `#include <cstdint>` in OpenVINO GPU plugin source for the L0 build path. Tiny OpenVINO GPU inference succeeded.

The llama.cpp OpenVINO backend is not currently competitive for Qwen3.6 27B GGUF on B70. The main issue is graph fragmentation: Qwen3.6 recurrent/Gated DeltaNet blocks split into hundreds of CPU/GPU islands, and compile/load time dominates. Basic sigmoid and softplus OpenVINO unary support helped split count but did not address the core recurrent path.

The clean llama.cpp patch currently worth carrying is in `patches/llama-b70-openvino-vulkan.patch`:

- fixes the oneAPI 2026 SYCL linker flag spelling for the greater-than-4GB buffer option;
- adds Intel PCI device `0xE223` as Arc Pro B70 with 32 shader cores to the Vulkan backend.

## Quality Notes

The over-40 tok/s numbers are from the INT4 AutoRound model, not from the Q4_0 GGUF and not from fp8/fp16. That is a real quantization tradeoff. It should be benchmarked with quality evals before using it as a quality-equivalent replacement.

The MTP speedup should not intentionally change accepted-token semantics, but the local XPU MTP fallback is still a patch over a code path vLLM marked unsupported on XPU. Treat it as an optimization result that needs evals, not as proven production quality.

## Next Work

- Submit and track all accepted LocalMaxxing result IDs.
- Run a quality sanity suite against INT4 AutoRound MTP versus non-MTP and GGUF Q4_0.
- Implement or prototype an XPU-native speculative Gated DeltaNet kernel path to replace the generic fallback.
- Investigate vLLM TP2 graph capture around XCCL communication instead of forcing CUDA graph assumptions.
- Test Mesa main ANV for B70 Vulkan once the dependency/build path is practical.
- Continue tracking Intel compute-runtime, IGC, OpenVINO, llama.cpp, and vLLM changes as B70 support matures.
