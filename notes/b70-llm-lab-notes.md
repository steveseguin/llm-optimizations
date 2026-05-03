# Intel Arc Pro B70 LLM Lab Notes

Date: 2026-05-03

## Current Best State

Important correction:

- The over-40 tok/s vLLM result is an INT4 AutoRound result and is not counted as success against the original quality-preserving Q4_0 GGUF target.
- The active Q4_0 plan is `plans/q4_0-gguf-b70-optimization-plan.md`.
- New Q4_0 benchmark harnesses:
  - `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`
  - `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`

Best single-B70 speed path so far:

- Engine: vLLM 0.20.1 XPU.
- Model: `Lorbus/Qwen3.6-27B-int4-AutoRound`.
- Patch: local XPU MTP fallback for Qwen3.6 Gated DeltaNet speculative metadata.
- Shape `input=500`, `output=256`: `5.67 s`, about `45.2 output tok/s`, `133.44 total tok/s`.
- Shape `input=500`, `output=512`: `12.40 s`, about `41.3 output tok/s`, `81.60 total tok/s`.
- Treat this as a separate quantization tradeoff result, not as Q4_0 GGUF success.

Best dual-B70 speed path so far:

- Engine: vLLM 0.20.1 XPU, tensor parallel size 2, non-MTP.
- Shape `input=500`, `output=256`: `5.22 s`, about `49.1 output tok/s`, `144.88 total tok/s`.
- Shape `input=500`, `output=512`: `10.59 s`, about `48.3 output tok/s`, `95.56 total tok/s`.
- TP2 is useful but still well below the desired 80% single-session uplift over the best single-card MTP path.

GGUF paths so far:

- llama.cpp SYCL single-card Qwen3.6 27B Q4_0 GGUF: about `24.6 tok/s` decode.
- llama.cpp Vulkan with B70 core-count patch on system Mesa: about `22 tok/s` decode.
- Windows Q4_0 comparison target is `>=27 tok/s`, with external results up to about `28.8 tok/s`.
- llama.cpp dual split is not viable locally yet.

## Q4_0 GGUF Continuation

- Decision: do not count the INT4 AutoRound result as meeting the Q4_0 GGUF target.
- Active plan: `plans/q4_0-gguf-b70-optimization-plan.md`.
- Added Q4_0 harnesses:
  - `scripts/bench-qwen36-q4_0-gguf-vulkan-matrix.sh`
  - `scripts/bench-qwen36-q4_0-gguf-sycl-matrix.sh`
- Sanity run through the Vulkan harness:
  - Local output: `/home/steve/bench-results/qwen36-q4_0-gguf/vulkan-20260503T191806Z.jsonl`
  - Command shape: Vulkan0, Q4_0 GGUF, `-p 0 -n 512`, `-fa 0`, `-ub 64`, `--poll 50`, compute queue, `-ctk f16 -ctv f16`, one rep.
  - Result: `22.19 tok/s`.
  - Interpretation: still system-Mesa baseline behavior; no Linux Q4_0 improvement yet.

## LocalMaxxing Submissions

All submitted vLLM INT4 results returned `APPROVED`.

| Label | LocalMaxxing ID | GPUs | Input | Output | tok/s out | tok/s total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm-int4-single-b70-mtp-500-256` | `cmoq41b9d001alg043wsnthz2` | 1 | 500 | 256 | 45.2 | 133.44 |
| `vllm-int4-single-b70-mtp-500-512` | `cmoq47sll0005l104v3i0f9l3` | 1 | 500 | 512 | 41.3 | 81.60 |
| `vllm-int4-tp2-b70-nonmtp-500-256` | `cmoq4e9dw0002js04ledqyycn` | 2 | 500 | 256 | 49.1 | 144.88 |
| `vllm-int4-tp2-b70-nonmtp-500-512` | `cmoq4krfb000cl40456wobg7e` | 2 | 500 | 512 | 48.3 | 95.56 |
| `vllm-int4-single-b70-nonmtp-500-256` | `cmoq4r8rc0001l804tocgibus` | 1 | 500 | 256 | 31.8 | 93.80 |
| `vllm-int4-tp2-b70-mtp-500-256` | `cmoq4xppt0003ky04xidngli9` | 2 | 500 | 256 | 35.6 | 105.03 |

## Hardware

- Host: Ubuntu 24.04 LTS, kernel 6.17.0-14-generic.
- CPU: AMD EPYC 9015.
- RAM reported for benchmark payloads: 16 GB.
- GPUs: 2x Intel Arc Pro B70 / BMG-G31, 32 GB each, exposed as `/dev/dri/renderD128` and `/dev/dri/renderD129`.
- Intel compute-runtime release installed: 26.14.37833.4.
- IGC: 2.32.7.
- Level Zero loader/tools: installed and visible to OpenVINO/SYCL.
- oneAPI compiler/MKL/DNNL 2026.0 installed.

## Latest Stack Signals

- Intel compute-runtime `26.14.37833.4` is still the latest GitHub release found on 2026-05-03 and is already installed.
- Local llama.cpp is at `b9010`/`d05fe1d`; `origin/master` is one commit ahead at `db44417`.
- Upstream llama.cpp still lacks the B70 Vulkan core-count entry for PCI ID `0xE223`, so the local patch remains required.
- OpenVINO docs include an internal `GatedDeltaNet` op specification, but local OpenVINO 2026.1.2 source does not expose a matching implementation under `src/`; this is now an explicit R&D item.

## XCCL P2P Microbenchmark

With `CCL_TOPO_P2P_ACCESS=1`:

- 4 KiB: 0.016 ms.
- 64 KiB: 0.014 ms.
- 1 MiB: 0.029 ms, about 35.9 GB/s.
- 16 MiB: 0.408 ms, about 41.2 GB/s.
- 64 MiB: 1.615 ms, about 41.6 GB/s.
- 256 MiB: 6.417 ms, about 41.8 GB/s.

With P2P disabled, large all-reduce falls to about 5 GB/s. Communication bandwidth is not the main TP2 blocker; vLLM's disabled XPU graph capture around communication is the larger issue.

## Quality Notes

The over-40 tok/s numbers are from the INT4 AutoRound model, not from the Q4_0 GGUF and not from fp8/fp16. That is a real quantization tradeoff. It should be benchmarked with quality evals before using it as a quality-equivalent replacement.

The MTP speedup should not intentionally change accepted-token semantics, but the local XPU MTP fallback is still a patch over a code path vLLM marked unsupported on XPU. Treat it as an optimization result that needs evals, not as proven production quality.

## Next Work

- Run the Q4_0 Vulkan and SYCL matrices and submit only comparable GGUF results to LocalMaxxing.
- Rebuild llama.cpp from current upstream plus the B70 `0xE223` Vulkan patch.
- Build/test Mesa main ANV locally and compare against system Mesa.
- Reproduce the Windows SYCL Q4_0 command shape and instrument Q4_0 reorder.
- Keep dual-GPU GGUF work paused until single-card Q4_0 is at least back in the Windows range.
