# Qwen3.6 27B Q4_0 GGUF SYCL Single-Kernel Allreduce Results

Date: 2026-05-04 UTC

Host: Ubuntu 24.04.4 LTS, AMD EPYC 9015, 4x Intel Arc Pro B70 / BMG-G31 32 GB. No GPU power-limit or clock changes were made.

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

llama.cpp worktree: `/home/steve/src/llama.cpp-q4-b70`, upstream `db44417` plus local experimental SYCL/Vulkan patches.

Runtime stack observed locally:

- Intel Compute Runtime: `26.14.37833.4`
- Level Zero runtime: `1.15.37833+4`
- OpenCL NEO: `26.14.37833.4`

## Key Patches

- `GGML_SYCL_ASYNC_CPY_TENSOR=1`: non-blocking SYCL backend tensor copy hook for Meta tensor-parallel activation exchange.
- Qwen recurrent split-anchor fix: recurrent `attn_qkv.*` and `attn_gate.weight` anchor to `ssm_out.weight` so 3-way tensor split uses a consistent split plan.
- `GGML_SYCL_COMM_ALLREDUCE=1`: SYCL backend Meta comm hook for contiguous F32 allreduces.
- `GGML_SYCL_COMM_SINGLE_KERNEL=1`: experimental 2-4 GPU root-kernel allreduce. Backend 0 peer-reads all partial F32 tensors and writes the summed hidden state back to every participating device.
- Vulkan side patch: recognize B70 PCI ID `0xE223` as 32 Xe2 shader cores and expose `GGML_VK_INTEL_XE2_DMMV_LARGE_MAX_M` for DMMV experiments.

The single-kernel allreduce remains env-gated because it is more aggressive than device-to-device copy plus local sum.

## Correctness Probe

Local stress test: `/home/steve/sycl-peer-read-test.cpp`.

Validation command:

```bash
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
icpx -O2 -fsycl /home/steve/sycl-peer-read-test.cpp -o /home/steve/sycl-peer-read-test
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 timeout 3m /home/steve/sycl-peer-read-test
```

Result:

```text
peer kernel read ok across 4 devices
```

The test writes all source tensors from device kernels, runs the same root-device peer-read/peer-write sum pattern used by the experimental allreduce, and verifies every device sees the exact expected F32 sum over 200 repeats.

## Current Best Commands

### 2 B70s

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,3 \
GGML_SYCL_DISABLE_GRAPH=0 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_DISABLE_OPT=0 \
GGML_SYCL_PRIORITIZE_DMMV=0 \
GGML_SYCL_ASYNC_PEER_COPY=0 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  --prio 0 -dev SYCL0/SYCL1 -ngl 99 -p 0 -n 512 \
  -sm tensor -ts 1/1 -b 512 -ub 32 -ctk f16 -ctv f16 \
  -t 8 --poll 50 -fa 1 -r 3 -o jsonl
```

Result: `39.849 tok/s`, samples `39.8517`, `39.7809`, `39.9148`.

LocalMaxxing ID: `cmoqp6jpq0004lb04241n9ns3`.

### 3 B70s

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3 \
GGML_SYCL_DISABLE_GRAPH=0 \
GGML_SYCL_DISABLE_DNN=1 \
GGML_SYCL_DISABLE_OPT=0 \
GGML_SYCL_PRIORITIZE_DMMV=0 \
GGML_SYCL_ASYNC_PEER_COPY=0 \
GGML_SYCL_ASYNC_CPY_TENSOR=1 \
GGML_SYCL_COMM_ALLREDUCE=1 \
GGML_SYCL_COMM_SINGLE_KERNEL=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -m /home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf \
  --prio 0 -dev SYCL0/SYCL1/SYCL2 -ngl 99 -p 0 -n 512 \
  -sm tensor -ts 1/1/1 -b 512 -ub 32 -ctk f16 -ctv f16 \
  -t 8 --poll 50 -fa 1 -r 3 -o jsonl
```

Result: `41.737 tok/s`, samples `41.6977`, `41.6966`, `41.8155`.

LocalMaxxing ID: `cmoqqed6s0007jv049wnizwle`.

## Results Summary

| Setup | Selector | Flags | 512-token tok/s | Notes |
| --- | --- | --- | ---: | --- |
| 1x B70 baseline | `0` | single device | `24.723` | Q4_0, f16 KV |
| 2x B70 async copy | `0,3` | `ASYNC_CPY_TENSOR=1` | `37.690` | LocalMaxxing `cmoqkcqpv0006la04l5mtlj2q` |
| 2x B70 direct allreduce | `0,3` | `COMM_ALLREDUCE=1` | `38.621` | LocalMaxxing `cmoqnkcx10006kw04f2jmahrv` |
| 2x B70 single-kernel allreduce | `0,3` | `COMM_SINGLE_KERNEL=1` | `39.849` | LocalMaxxing `cmoqp6jpq0004lb04241n9ns3` |
| 3x B70 split-anchor only | `0,1,2` | async copy, split fix | `38.365` | LocalMaxxing `cmoqli4dm0005l404kdf9ofnd` |
| 3x B70 single-kernel allreduce | `0,1,2` | root 0 | `41.367` | LocalMaxxing `cmoqptj6i000blb04j0i2u9yo` |
| 3x B70 single-kernel allreduce | `2,1,3` | root 2 | `41.737` | Current best, LocalMaxxing `cmoqqed6s0007jv049wnizwle` |
| 4x B70 single-kernel allreduce | `0,1,2,3` | root 0 | `31.482` | Still not viable for single-session speed |

## Follow-Up Sweeps

3-GPU root-order sweep, 128 tokens:

- `2,1,3`: `41.358 tok/s`
- `0,1,2`: `41.129 tok/s`
- `1,2,3`: `41.075 tok/s`
- `3,0,2`: `41.032 tok/s`

3-GPU `-ub` sweep:

- Short-run `-ub 128` won, but full 512-token validation fell to `41.113 tok/s`.
- Keep `-ub 32`.

3-GPU `--poll` sweep:

- `--poll 50` remains best.

4-GPU root-order sweep:

- Best root order `3,0,1,2`: `31.311 tok/s` at 128 tokens.
- All tested root orders stayed near `30.76-31.31 tok/s`, so 4-way weakness is communication fanout/synchronization, not root-card choice.

## Single-Card Follow-Up

Focused single-card flag sweep:

- Output summary: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-single-selector2-dnn-fa-ub-sweep-n256-20260504T052313Z.tsv`.
- Shape: selector `2`, `-dev SYCL0`, `-sm none`, 256 generated tokens, f16 KV, no speculative decode.
- Best short result: oneDNN enabled, flash attention disabled, `-ub 128`: `24.449 tok/s`.
- Flash attention was slightly slower in this decode-only shape.
- oneDNN on/off and `-ub 32/64/128/256` did not move the result meaningfully.

Build comparison:

- Output summary: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-single-build-compare-n256-20260504T053253Z.tsv`.
- `aot-dnn`: `24.805 tok/s`.
- `dnn`: `24.756 tok/s`.
- `aot`: `24.561 tok/s`.
- current base build: `24.498 tok/s`.

Conclusion: the Linux single-card Q4_0 path is still below the Windows `>27 tok/s` target. The gap is not explained by oneDNN, flash attention, ubatch, or the current AOT build variants; next single-card work should profile the Q4_0 reordered MMVQ/matvec path and driver/runtime scheduling.

## Current Read

The best quality-preserving GGUF/SYCL single-session setup is three B70s, equal tensor split, selector order `2,1,3`, `-ub 32`, `--poll 50`, and the env-gated single-kernel allreduce path.

The fourth B70 is currently better used for a parallel session, a draft/speculative model, or further communication R&D. The remaining blocker is not raw model quality or power; it is the 128 small F32 allreduces per generated token and their synchronization cost.
