# 2026-05-06 Q4_0 reordered MMVQ VDR and XOR-reduce screen

## Goal

Check whether Qwen3.6 27B Q4_0 GGUF decode speed on Arc Pro B70 improves by changing only the SYCL reordered MMVQ lane/block scheduling or subgroup reduction method. This preserves the Q4_0 model and math path; no quantization or output-quality tradeoff is introduced.

## Patch

Local patch:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-q4-vdr-xor-negative-20260506.patch`
- sha256: `d11f5617a9f6c718d9ccd4d8ec4ebd2f7ccdc4a1260209f5e7b92305b92068d9`

Runtime controls added:

- `GGML_SYCL_REORDER_MMVQ_Q4_VDR=1|4`
- `GGML_SYCL_REORDER_MMVQ_XOR_REDUCE=1`

Default remains the existing VDR=2 plus `sycl::reduce_over_group` behavior.

## Benchmark Setup

Model:

- `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

Hardware/runtime:

- 1x Intel Arc Pro B70 32GB
- selector `ONEAPI_DEVICE_SELECTOR=level_zero:2`
- llama.cpp SYCL AOT BMG G31 build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- decode-only short screen: `-p 0 -n 128 -r 2`
- fast-path envs: `GGML_SYCL_DISABLE_DNN=1`, `GGML_SYCL_Q8_CACHE=1`, `GGML_SYCL_FUSE_MMVQ2=1`, `GGML_SYCL_ASYNC_CPY_TENSOR=0`

## Results

| Variant | Decode tok/s | JSONL |
| --- | ---: | --- |
| default VDR=2 | 24.654646 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q4-vdrdefault-single2-p0n128-r2-20260506T110504Z.jsonl` |
| VDR=1 | 24.516621 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q4-vdr1-single2-p0n128-r2-20260506T110504Z.jsonl` |
| VDR=4 | 23.232102 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q4-vdr4-single2-p0n128-r2-20260506T110504Z.jsonl` |
| default reduce | 24.599359 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q4-reduce-default-single2-p0n128-r2-20260506T111513Z.jsonl` |
| XOR reduce | 24.557468 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-q4-reduce-xor-single2-p0n128-r2-20260506T111513Z.jsonl` |

## Conclusion

Do not pursue this path as an optimization. VDR=1 and VDR=4 did not beat the existing VDR=2 scheduling, and explicit XOR subgroup reduction did not beat `sycl::reduce_over_group`.

This result is still useful because it narrows the single-card Q4_0 bottleneck away from this lane grouping/reduction choice. Next Q4 work should focus on either a deeper ESIMD/XMX Q4_0 x Q8_1 matvec implementation or graph-level communication/partial-output fusion for multi-GPU scaling.

LocalMaxxing: not submitted because this is a negative microbenchmark, not a shareable improved model run.
