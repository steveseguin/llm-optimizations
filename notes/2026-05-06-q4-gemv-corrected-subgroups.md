# 2026-05-06 Q4_0 GEMV harness correction and subgroup sweep

## Summary

- Corrected the standalone Q4_0 x Q8_1 GEMV harness to use llama/GGML Q4_0 packing semantics.
- Previous ESIMD prototype correctness was only valid against an incorrect adjacent low/high nibble reference. GGML Q4_0 stores low nibbles for positions `0..15` and high nibbles for positions `16..31`.
- Added `--mode=sgdp4a`, a production-style subgroup baseline that mirrors llama.cpp reordered MMVQ dispatch more closely than the first ESIMD float prototype.
- Rebuilt a llama.cpp test binary with `-DGGML_SYCL_REORDER_MMVQ_SUBGROUPS=4`; it is correctness-neutral on 1x but aborts during multi-GPU context reservation, so it is not a candidate yet.

## Harness Changes

- File: `/home/steve/llm-optimization-artifacts/tools/q4_0_gemv_esimd_bench.cpp`
- Added arguments:
  - `--mode=esimd|sgdp4a`
  - `--subgroups=1|2|4|8|16|32`
  - `--warp=16|32`
- Fixed CPU reference and ESIMD path:
  - old reference paired `lo -> q8[2*j]`, `hi -> q8[2*j+1]`
  - corrected reference pairs `lo -> q8[j]`, `hi -> q8[16+j]`
- Added production-style `sgdp4a` kernel:
  - one subgroup computes one row
  - `VDR=2`, `QI4_0=4`
  - subgroup reduction over partial Q4_0 x Q8_1 dot products
  - selectable subgroup width; llama.cpp's Intel SYCL build uses `GGML_SYCL_WARP_SIZE=16`

## Key Results

Corrected warmed cross-check, `N=17408 K=5120`, diagnostic `warp=32`:

- `sgdp4a`, subgroups 16: `92.088430 us` best warmed repeat, `545.241481 GB/s`
- `sgdp4a`, subgroups 4: `92.344330 us` repeat, first warmed run `86.616580 us`
- `esimd`, `ks=1`: `128.440435 us` then `144.448370 us`

Production-width subgroup sweep highlights, `warp=16`:

- `N=17408 K=5120`: best `subgroups=4`, `90.872058 us`; default `subgroups=16`, `97.747108 us`; `subgroups=32`, `99.651375 us`
- `N=5120 K=17408`: best `subgroups=2`, `90.435475 us`; default `subgroups=16`, `92.707742 us`; `subgroups=32`, `93.571325 us`
- `N=4352 K=5120`: best `subgroups=1`, `22.460742 us`; default `subgroups=16`, `26.529775 us`
- `N=5120 K=4352`: best `subgroups=16`, `26.813025 us`
- `N=5120 K=1536`: best `subgroups=2`, `7.929633 us`

The subgroup baseline is faster than the corrected ESIMD float prototype on the major FFN shapes, so the next Q4 kernel work should not integrate the current ESIMD path. A useful ESIMD follow-up would need packed integer dot products or XMX/DPAS-style work, not float unpack/FMA.

## llama.cpp sg4 Build Test

- Build dir: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-sg4`
- Configure delta: `-DGGML_SYCL_REORDER_MMVQ_SUBGROUPS=4` with the normal Intel SYCL `GGML_SYCL_WARP_SIZE=16`
- 1x deterministic probe matched the default build exactly for 32 generated tokens:
  - `/home/steve/bench-results/qwen36-q4_0-gguf/correctness/sg4-subgroups-20260506T093748Z`
- 1x 512/512 benchmark:
  - `24.659991 tok/s`, neutral/slightly below prior default `24.723 tok/s`
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-subgroups4-sg4-single-p512n512-r3-20260506T093926Z.jsonl`
- Multi-GPU sg4 with correct slash device syntax aborted before token generation:
  - assert: `ggml_backend.cpp:120: GGML_ASSERT(buffer) failed`
  - backtrace enters `ggml_backend_meta_buffer_type_alloc_buffer` during context reserve
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-subgroups4-sg4-triple213-p512n512-r3-20260506T094515Z.log`
- Default build with the same slash syntax works:
  - `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-default-devslash-triple213-smoke-20260506T094636Z.jsonl`

## Decision

- Keep the production Q4 path on the existing default build for now.
- Treat the prior ESIMD prototype as a corrected harness, not an integration candidate.
- Do not submit sg4 to LocalMaxxing: it is not faster on 1x and it is not valid on multi-GPU.
- Next kernel work should focus on either packed integer ESIMD/XMX or graph-level fusion around `ffn_gate + ffn_up + swiglu`, not the current float ESIMD GEMV.
- If revisiting subgroup tuning, debug the sg4 multi-GPU context-reserve abort first; the harness suggests lower subgroup counts can improve some production-width GEMV shapes, but the compiled sg4 binary is not valid on tensor split yet.
