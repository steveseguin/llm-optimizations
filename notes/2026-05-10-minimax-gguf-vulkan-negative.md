# MiniMax GGUF Vulkan Backend Smoke, 2026-05-10

Target: `unsloth/MiniMax-M2.7-GGUF` `UD-IQ4_XS`, local sharded GGUF under
`/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS`, llama.cpp Vulkan
build at `/home/steve/src/llama.cpp/build-vulkan-b70/bin/llama-bench`, 4x
Intel Arc Pro B70, no power-limit changes.

## Why

The existing GGUF path uses ik_llama.cpp RPC+SYCL process-per-GPU layer split
and tops out around `17.697772` tok/s. Vulkan is a separate llama.cpp backend
with all four B70s visible as `Vulkan0..Vulkan3`, so it was worth checking
whether it could avoid the SYCL/RPC sequential-layer bottleneck.

## System Stack Notes

- Vulkan driver: Mesa `25.2.8-0ubuntu0.24.04.1`, API `1.4.318`.
- Compute stack installed locally:
  - `intel-opencl-icd 26.14.37833.4-0`
  - `libze-intel-gpu1 26.14.37833.4-0`
  - `intel-ocloc 26.14.37833.4-0`
  - `intel-igc-core-2 2.32.7`
  - `intel-igc-opencl-2 2.32.7`
  - `level-zero 1.28.2`
  - `libigdgmm12 22.9.0`
- Upstream check, 2026-05-10:
  - Intel compute-runtime GitHub release page lists `26.05.37020.3` and notes
    Battlemage validation on Ubuntu 24.04 with the intel-graphics PPA kernel:
    <https://github.com/intel/compute-runtime/releases>
  - Intel OpenVINO 2026.1 lists the llama.cpp backend as preview GGUF support:
    <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/whats-new.html>
  - Intel oneDNN 2026 release notes call out Xe2 fp16 matmul regression and
    Arc B-series f32 matmul correctness risk, which argues against chasing
    oneDNN toggles for this MiniMax path:
    <https://www.intel.com/content/www/us/en/developer/articles/release-notes/onednn/2026.html>

## Results

All runs used:

```bash
/home/steve/src/llama.cpp/build-vulkan-b70/bin/llama-bench \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev Vulkan0/Vulkan1/Vulkan2/Vulkan3 \
  -p 0 -n 16 -r 1 -ngl 99 -ts 1/1/1/1 -fa 0 -nkvo 0 \
  -ub 64 -ctk f16 -ctv f16 -t 4 -o jsonl
```

| Split mode | Result |
| --- | ---: |
| `-sm layer` | `11.745588` tok/s |
| `-sm row` | `12.558578` tok/s |
| `-sm tensor` | failed at model load |

Artifacts:

- Layer log: `/mnt/fast-ai/bench-results/minimax-m2.7-gguf-vulkan/minimax-m27-ud-iq4xs-vulkan4-p0n16-20260510T115154Z.log`
- Layer JSONL: `/mnt/fast-ai/bench-results/minimax-m2.7-gguf-vulkan/minimax-m27-ud-iq4xs-vulkan4-p0n16-20260510T115154Z.jsonl`
- Row log: `/mnt/fast-ai/bench-results/minimax-m2.7-gguf-vulkan/minimax-m27-ud-iq4xs-vulkan4-row-p0n16-20260510T115418Z.log`
- Row JSONL: `/mnt/fast-ai/bench-results/minimax-m2.7-gguf-vulkan/minimax-m27-ud-iq4xs-vulkan4-row-p0n16-20260510T115418Z.jsonl`
- Tensor-fail log: `/mnt/fast-ai/bench-results/minimax-m2.7-gguf-vulkan/minimax-m27-ud-iq4xs-vulkan4-tensor-p0n16-20260510T115635Z.log`

## Conclusion

Do not pursue Vulkan as the current MiniMax GGUF speed path. It is functional
for layer and row split but slower than the existing SYCL/RPC GGUF route, and
tensor split does not load this model. The >30 tok/s MiniMax work should stay
with the vLLM/AutoRound path or require a deeper quality-correct GGUF
graph/tensor-parallel rewrite.
