# 2026-05-06 Q4_0 B70 pair/triple/quad topology screen

## Goal

Explain why Qwen3.6 27B Q4_0 GGUF scales well from one to three B70s but falls back on four B70s. This screen checks whether the failure is tied to one bad GPU, one bad pair, root ordering, or the 4-way tensor-parallel path itself.

## Benchmark Setup

Model:

- `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

Common command shape:

- llama.cpp SYCL AOT BMG G31 build: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- tensor split mode
- `-p 512 -n 128 -r 2`
- `-fa 1 -ub 32 -ctk f16 -ctv f16 -ngl 99`
- `GGML_SYCL_DISABLE_DNN=1`
- `GGML_SYCL_Q8_CACHE=1`
- `GGML_SYCL_ASYNC_CPY_TENSOR=0`
- `GGML_SYCL_ASYNC_PEER_COPY=1`
- `GGML_SYCL_COMM_ALLREDUCE=1`
- `GGML_SYCL_COMM_SINGLE_KERNEL=1`
- `GGML_SYCL_COMM_EVENT_BARRIER=1`
- `GGML_SYCL_COMM_SYNC_AFTER=2`
- `GGML_META_FUSE_ALLREDUCE_ADD=1`
- `GGML_SYCL_FUSE_MMVQ2=1`

## Results

Pair sweep summary:

- all 12 ordered two-card pairs landed tightly at `40.687815` to `40.747873 tok/s`;
- no pair or root ordering stood out as bad;
- summary: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-pair-order-sweep-devslash-p512n128-r2-20260506T113813Z.jsonl`.

Triple/quad sweep:

| Selector | GPU count | Decode tok/s | JSONL |
| --- | ---: | ---: | --- |
| `2,1,3` | 3 | 45.195064 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq2_1_3-p512n128-r2-20260506T115036Z.jsonl` |
| `0,1,2` | 3 | 45.415802 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq0_1_2-p512n128-r2-20260506T115036Z.jsonl` |
| `0,1,3` | 3 | 45.140951 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq0_1_3-p512n128-r2-20260506T115036Z.jsonl` |
| `0,2,3` | 3 | 46.037050 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq0_2_3-p512n128-r2-20260506T115036Z.jsonl` |
| `2,1,3,0` | 4 | 34.550830 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq2_1_3_0-p512n128-r2-20260506T115036Z.jsonl` |
| `0,1,2,3` | 4 | 34.722804 | `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-tq0_1_2_3-p512n128-r2-20260506T115036Z.jsonl` |

Triple/quad summary:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-triple-quad-order-sweep-p512n128-r2-20260506T115036Z.jsonl`

## Conclusion

The 4-card regression is not a bad B70, a bad pair, or a simple root-ordering problem. Every ordered pair is healthy, and every tested triple is healthy, including triples with device 0. The collapse only appears at four devices.

The next Q4_0 multi-GPU work should focus on the 4-way allreduce/scheduling implementation, especially whether the single-kernel allreduce path has a 4-participant occupancy, barrier, or peer-copy serialization problem. A useful next test is an env-gated 4-way pairwise/tree reduce that keeps the current 2-card/3-card paths unchanged.

LocalMaxxing: not submitted. This is a diagnostic topology screen, and the already submitted 3-card and 4-card full 512/512 runs are better leaderboard-quality records.
