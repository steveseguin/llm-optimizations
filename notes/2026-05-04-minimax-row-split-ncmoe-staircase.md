# MiniMax M2.7 Row-Split `-ncmoe` Staircase

Date: 2026-05-04

## Context

MiniMax M2.7 UD-IQ4_XS is present locally as four GGUF shards under:

`/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS`

The goal was to see whether the 101 GB GGUF could run across four 32 GB B70s. Tensor split is not implemented for `minimax-m2` in this llama.cpp tree, and layer split fails on large contiguous device allocations. Row split gets further, but reaches expert split-buffer allocation failures.

## Command Shape

```bash
source /opt/intel/oneapi/setvars.sh --force

ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
GGML_SYCL_DISABLE_DNN=1 \
/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31/bin/llama-bench \
  -v \
  -m /home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -dev SYCL0/SYCL1/SYCL2/SYCL3 \
  -ngl 99 \
  -sm row \
  -ts 1/1/1/1 \
  -ncmoe <N> \
  -p 0 \
  -n 1 \
  -b 128 \
  -ub 32 \
  -fa 0 \
  -ctk f16 \
  -ctv f16 \
  -t 8 \
  -r 1 \
  -o jsonl \
  -oe jsonl
```

## Results

| `-ncmoe` | Outcome | First Failed GPU Tensor | Device | Allocation |
| ---: | --- | --- | ---: | ---: |
| `13` | load failed | `blk.25.ffn_gate_exps.weight`, `iq3_s`, rows `[98304, 196608)` | `1` | `129761280` bytes |
| `26` | load failed | `blk.37.ffn_up_exps.weight`, `iq3_s`, rows `[0, 98304)` | `0` | `129761280` bytes |
| `38` | load failed | `blk.49.ffn_gate_exps.weight`, `iq3_s`, rows `[98304, 196608)` | `1` | `129761280` bytes |
| `50` | load failed | `blk.60.ffn_up_exps.weight`, `iq3_s`, rows `[0, 98304)` | `0` | `129761280` bytes |

Logs:

- `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe13-verbose-20260504T224723Z.log`
- `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe26-verbose-20260504T224846Z.log`
- `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe38-verbose-20260504T225001Z.log`
- `/home/steve/bench-results/minimax-m2.7-ud-iq4_xs-gguf/minimax-m27-quad-row-ngl99-ncmoe50-verbose-20260504T225119Z.log`

## Interpretation

The failure moves forward by about 12 GPU-resident expert layers each time more experts are forced to host with `-ncmoe`. That makes this look like cumulative split-buffer allocation pressure or fragmentation in the SYCL row-split path, not a single bad tensor or bad row range.

`-ncmoe 62` can load because it effectively pushes all experts to CPU/file-backed memory, but that is not a useful speed path on a 15 GB RAM host. No MiniMax result should be submitted to LocalMaxxing yet.

## Next Work

Stop treating MiniMax as a flag sweep. The useful implementation boundary is now:

- split-buffer allocation behavior for expert tensors;
- `GGML_OP_MUL_MAT_ID` execution with SYCL split buffers;
- or an expert-aligned path where selected experts run on the owning B70 and outputs are assembled correctly.

`--no-mmap` is not a realistic workaround on this host because the model is about 101 GB and system RAM is only about 15 GB.
