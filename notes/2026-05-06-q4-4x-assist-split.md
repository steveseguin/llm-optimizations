# 2026-05-06 Qwen3.6 Q4_0 4x Assist Split

## Summary

After reboot, the equal four-card Qwen3.6 27B Q4_0 GGUF tensor split still reproduced the known negative scaling range. A targeted tensor-split sweep showed that the fourth B70 is only useful as a small assist device on the current llama.cpp/SYCL path.

Best validated four-card result:

- Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`
- Quantization: `Q4_0`
- KV cache: `f16`
- Flash attention: enabled
- Speculative decoding: disabled
- GPU power changes: none
- Selector order: `level_zero:2,1,3,0`
- Devices: `SYCL0/SYCL1/SYCL2/SYCL3`
- Tensor split: `1/1/1/0.05`
- Runtime subgroup override: `GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2`
- Prompt/decode validation: `-p 512 -n 512 -r 3`
- Prompt throughput: `80.906412 tok/s`
- Decode throughput: `39.204149 tok/s`
- Total throughput: `52.815789 tok/s`
- LocalMaxxing ID: `cmou581wv002dld0197mffpco`

This improves the previous full four-card equal-split validation from `34.929313 tok/s` to `39.204149 tok/s`, a `12.24%` decode gain. It is still below the best three-card Q4_0 result of `46.194319 tok/s`, so four-way Q4_0 tensor parallelism remains a diagnostic path rather than the best production path.

## Command

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3,0 \
GGML_SYCL_DISABLE_GRAPH=0 \
GGML_SYCL_Q8_CACHE=1 \
GGML_SYCL_FUSE_MMVQ2=1 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
GGML_SYCL_COMM_EVENT_BARRIER=1 \
GGML_SYCL_COMM_SYNC_AFTER=2 \
GGML_META_FUSE_ALLREDUCE_ADD=1 \
GGML_SYCL_REORDER_MMVQ_SUBGROUPS_RUNTIME=2 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  -ngl 99 -fa 1 -ctk f16 -ctv f16 \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -sm tensor -ts 1/1/1/0.05 \
  -b 2048 -ub 32 --poll 50 -t 8 \
  -p 512 -n 512 -r 3 -o jsonl
```

## Screening Runs

Runtime subgroup sweep, four-card equal split:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/subgroup-runtime-quad2130-sweep-p0n128-r2-20260506T134331Z.tsv`
- default: `34.091961 tok/s`
- subgroup runtime `1`: `34.445365 tok/s`
- subgroup runtime `2`: `34.910050 tok/s`
- subgroup runtime `4`: `34.483940 tok/s`
- subgroup runtime `8`: `34.906615 tok/s`
- subgroup runtime `16`: `34.332579 tok/s`

Four-card split sweep with subgroup runtime `2`:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/tensorsplit-quad-sg2-short-p0n128-r2-20260506T135139Z.tsv`
- `1/1/1/1`: `34.388387 tok/s`
- `1/1/1/0.75`: `35.414166 tok/s`
- `1/1/1/0.5`: `34.891262 tok/s`
- `1/1/1/0.25`: `36.735744 tok/s`

Fourth-card assist-ratio sweep:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/tensorsplit-quad-sg2-assist-ratio-p0n128-r2-20260506T140456Z.tsv`
- `1/1/1/0.05`: `38.920367 tok/s`
- `1/1/1/0.10`: `38.755095 tok/s`
- `1/1/1/0.15`: `38.831500 tok/s`
- `1/1/1/0.20`: `37.150397 tok/s`
- `1/1/1/0.25`: `36.369368 tok/s`
- `1/1/1/0.30`: `36.498531 tok/s`
- `1/1/1/0.40`: `36.435045 tok/s`
- `1/1/1/0.60`: `35.307680 tok/s`

## Interpretation

The fourth card is not bad; prior topology sweeps showed all pairs and triples are healthy. The current four-way regression is a software scheduling/kernel-shape problem. Equal four-way row shards appear too narrow for the current reordered Q4_0 MMVQ path, and the added communication/synchronization overhead outweighs the reduced work per card.

The assist split improves four-card use without changing quality, but it is still not true four-card scaling. Next high-value Q4 work should target:

- a Q4_0 MMVQ kernel path that remains efficient for narrower four-way shards;
- output-projection plus allreduce/residual epilogue fusion;
- lower collective frequency or fewer decode graph synchronization points.

FP8/vLLM remains the stronger all-four-card path today, with the caveat that it is a different model/backend track rather than an apples-to-apples Q4_0 GGUF result.
