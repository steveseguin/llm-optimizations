# 2026-05-05 Post-Reboot Qwen3.6 Q4_0 Reshape-ADD 3x/4x Results

After a failed PCI function reset attempt wedged xe/Level Zero, a full reboot restored the B70 runtime. Do not use FLR/PCI reset as a B70 recovery method on this driver stack.

Clean recovery checks:

- `sycl-ls`: four Level Zero B70 devices visible.
- `/home/steve/sycl-peer-read-test`: `peer kernel read ok across 4 devices`.
- Kernel: `6.17.0-23-generic`.
- GuC: `xe/bmg_guc_70.bin` `70.49.4`.
- dmesg after benchmark checks: no new GPU reset/AER/error lines beyond boot noise.

## 3x Current Control

Qwen3.6 27B Q4_0 GGUF, llama.cpp SYCL Level Zero, selector `2,1,3`, tensor split `1/1/1`.

Environment:

```text
GGML_SYCL_DISABLE_DNN=1
GGML_SYCL_Q8_CACHE=1
GGML_SYCL_ASYNC_CPY_TENSOR=1
GGML_SYCL_COMM_ALLREDUCE=1
GGML_SYCL_COMM_SINGLE_KERNEL=1
GGML_SYCL_COMM_EVENT_BARRIER=1
GGML_META_FUSE_ALLREDUCE_ADD=1
```

Command:

```text
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf -dev SYCL0/SYCL1/SYCL2 -ngl 99 -sm tensor -ts 1/1/1 -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 3 --poll 50 -o jsonl
```

Result:

- prompt: `135.469357 tok/s`;
- decode: `45.624065 tok/s`;
- decode samples: `45.5129`, `45.4826`, `45.8768`;
- computed total: `68.259384 tok/s`;
- LocalMaxxing: `cmot9sgsi000lib042rqd6c62`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-postreboot-reshapeadd-triple213-p512n512-r3-20260505T233641Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-postreboot-reshapeadd-triple213-p512n512-r3-20260505T233641Z.log`.

Quality note: same Q4_0 GGUF, f16 KV, no speculative decode, no sampling change, no GPU power-limit change. The speed path is graph scheduling plus SYCL communication helpers.

## 4x Negative Scaling

Same quality-preserving path, selector `0,1,2,3`, tensor split `1/1/1/1`.

Command:

```text
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf -dev SYCL0/SYCL1/SYCL2/SYCL3 -ngl 99 -sm tensor -ts 1/1/1/1 -fa 1 -ub 32 -ctk f16 -ctv f16 -t 8 -p 512 -n 512 -r 1 --poll 50 -o jsonl
```

Result:

- prompt: `102.210613 tok/s`;
- decode: `34.375523 tok/s`;
- computed total: `51.448022 tok/s`;
- LocalMaxxing diagnostic: `cmota1fpx0001l404wepbjtb7`;
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-postreboot-reshapeadd-quad0123-p512n512-r1-20260505T234410Z.jsonl`;
- log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-k617023-postreboot-reshapeadd-quad0123-p512n512-r1-20260505T234410Z.log`.

Interpretation: 4x is stable from a clean boot, but slower than 3x for single-session Q4_0. The fourth B70 currently adds more collective latency than it removes from row-parallel matvec work.

## Next

- Keep 3x `45.624 tok/s` as the current Q4_0 control.
- Inspect the final remaining plain allreduce, `attn_output-63 -> GET_ROWS`.
- Avoid more 4x root/order sweeps until the reduction count or allreduce implementation changes.
- Do not use PCI function reset as a B70 recovery method.
