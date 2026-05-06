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

Fine assist-ratio sweep after the validated `0.05` run:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/tensorsplit-quad-sg2-fine-assist-ratio-p0n128-r2-20260506T143835Z.tsv`
- `1/1/1/0.01`: failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during `MUL_MAT`
- `1/1/1/0.02`: failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during `MUL_MAT`
- `1/1/1/0.03`: `38.822236 tok/s`
- `1/1/1/0.04`: `38.135548 tok/s`
- `1/1/1/0.05`: `38.270868 tok/s`
- `1/1/1/0.06`: `37.856266 tok/s`
- `1/1/1/0.07`: `38.485052 tok/s`
- `1/1/1/0.08`: `38.311132 tok/s`
- `1/1/1/0.12`: `37.762326 tok/s`

This finer sweep did not beat the previous short `0.05` screen or the full validated `39.204149 tok/s` run. The lower `0.01` and `0.02` ratios are not just slower; they can produce a driver/runtime OOM in this path, likely because extremely small shards interact badly with tensor-split allocation or graph planning.

## Allreduce Probe

One-token allreduce stats on the four-card assist split:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-stats4-assist005-quad2130-p0n1-r1-20260506T145113Z.jsonl`
- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-stats4-assist005-quad2130-p0n1-r1-20260506T145113Z.log`
- Decode throughput under stats instrumentation: `24.699805 tok/s`
- Allreduce count: `128` per token
- Allreduce size: `20,480` bytes each
- Total allreduce payload: `2,621,440` bytes per token
- Warm stats summary: `4.213 ms` total allreduce time per token, `32.913 us` average
- First/cold stats summary: `6.724 ms` total allreduce time per token, `52.528 us` average

Every decoded token still performs two partial-output reductions per layer, mostly fused as `backend+add`. The fusion removes follow-on residual ADD scheduling, but it does not reduce the collective count. On the current 4x assist split, the warm collective cost is roughly one-sixth of the measured token budget, so communication remains worth optimizing.

Fused Q4 MMVQ2 status:

- Debug JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-fused2-triple213-p0n1-r1-20260506T144944Z.jsonl`
- Debug log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-fused2-triple213-p0n1-r1-20260506T144944Z.log`
- Debug counts: `480` fused2 calls and `1,584` plain reordered MMVQ calls

The fused2 kernel path is active in tensor split, including FFN gate/up pairs and some V/K pairs. Further Q4 work should focus on communication and narrow-shard MMVQ efficiency, not on merely enabling fused2.

## Communication Flag Sweep

Existing allreduce-path variants on the four-card assist split:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/comm-flag-sweep-quad-assist005-p0n128-r2-20260506T145544Z.tsv`
- Baseline `sync_after=2`: `38.195956 tok/s`
- `sync_after=0`: `38.354083 tok/s`
- `sync_after=1`: `37.158215 tok/s`
- `skip_root_ready`: `37.971362 tok/s`
- `rotate_root`: `37.087460 tok/s`
- `fuseadd_root_residual`: `37.898128 tok/s`
- `skiproot_fuseaddroot`: `38.218925 tok/s`
- `pairwise4`: `28.156880 tok/s`
- `striped4`: `26.468293 tok/s`
- `local_write`: `37.208619 tok/s`
- `no_fuseadd_smallf32`: `28.999878 tok/s`

No existing communication flag meaningfully beat the current single-kernel allreduce-add path. The tiny `sync_after=0` lift is within short-run noise and should not replace the validated command without a longer run. Pairwise, striped, and non-fused small-f32 paths are clearly worse for this 20 KB per-layer reduction shape.

## Narrow-Shard Follow-Ups

Additional diagnostics are captured in:

- Note: `/home/steve/llm-optimization-artifacts/notes/2026-05-06-q4-narrow-shard-followups.md`
- Data: `/home/steve/llm-optimization-artifacts/data/qwen36-q4-narrow-shard-followups-20260506.json`
- Patch: `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-split-skip-last-below-rows-focused-20260506.patch`

Key results:

- `GGML_SYCL_MMV_Y=2` was neutral on 3x and did not help 4x assist: `38.181852 tok/s`.
- MUL_MAT stage timing shows 4x assist adds `282` quantization calls and `538` matmul kernel launches versus 3x for essentially the same total matmul byte volume.
- Explicit `-ts 1/1/1/0` aborts at `ggml-backend.cpp:120: GGML_ASSERT(buffer) failed`; trailing-zero split is currently invalid.
- The env-gated `GGML_SYCL_SPLIT_SKIP_LAST_BELOW_ROWS` patch is safe when unset and passes a 3x sanity run at `44.778557 tok/s`, but threshold sweeps topped out at `38.153344 tok/s`, below the validated `39.204149 tok/s` 4x assist run.

Decision: keep the skip-last patch only as a diagnostic tool and leave the env unset for production. The fourth-card Q4_0 issue is not solved by simple row-threshold pruning.

## Interpretation

The fourth card is not bad; prior topology sweeps showed all pairs and triples are healthy. The current four-way regression is a software scheduling/kernel-shape problem. Equal four-way row shards appear too narrow for the current reordered Q4_0 MMVQ path, and the added communication/synchronization overhead outweighs the reduced work per card.

The assist split improves four-card use without changing quality, but it is still not true four-card scaling. Next high-value Q4 work should target:

- a Q4_0 MMVQ kernel path that remains efficient for narrower four-way shards;
- output-projection plus allreduce/residual epilogue fusion;
- lower collective frequency or fewer decode graph synchronization points.

FP8/vLLM remains the stronger all-four-card path today, with the caveat that it is a different model/backend track rather than an apples-to-apples Q4_0 GGUF result.
