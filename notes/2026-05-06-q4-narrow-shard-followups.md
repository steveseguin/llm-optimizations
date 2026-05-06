# 2026-05-06 Q4_0 Narrow-Shard Follow-Ups

## Summary

Follow-up probes after the four-card assist split did not find a better four-card Q4_0 path. The useful finding is diagnostic: adding a tiny fourth shard increases per-token quantization and kernel-launch work while barely changing total Q4 kernel bytes.

The production-quality Q4_0 path remains the three-B70 tensor split. Four-B70 Q4_0 remains a source-level investigation until the fourth device can do useful work without adding narrow-shard overhead.

## MMV_Y Probe

Separate build:

- Build path: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-mmv-y2`
- Compile flag: `-DGGML_SYCL_MMV_Y=2`
- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/mmv-y2-probe-p0n128-r2-20260506T152341Z.tsv`

Results:

- Current `MMV_Y=1`, 3x: `44.321101 tok/s`
- Experimental `MMV_Y=2`, 3x: `44.366675 tok/s`
- Experimental `MMV_Y=2`, 4x assist `1/1/1/0.05`: `38.181852 tok/s`

Decision: `MMV_Y=2` is effectively neutral on 3x and does not help 4x. Keep the default build for validated runs.

## MUL_MAT Stage Timing

Instrumentation:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/mulmat-stage-current-3x-vs-4x-p0n1-r1-20260506T152722Z.tsv`
- 3x log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mulstat-current_3x-p0n1-r1-20260506T152722Z.log`
- 4x assist log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-mulstat-current_4x_assist005-p0n1-r1-20260506T152722Z.log`

Final cumulative stats:

- 3x `quantize_main`: `1206` calls, `544.447 ms`, `24,436,736` bytes
- 3x `mul_mat_kernel`: `2982` calls, `424.418 ms`, `30,180,034,304` bytes
- 4x assist `quantize_main`: `1488` calls, `685.476 ms`, `27,754,496` bytes
- 4x assist `mul_mat_kernel`: `3520` calls, `518.254 ms`, `30,186,259,456` bytes

Interpretation: the fourth assist shard adds `282` quantization calls and `538` extra matmul kernel launches for essentially the same total matmul byte volume. This explains why the fourth B70 does not improve Q4_0 single-session speed on the current row-sharded MMVQ path.

The stage timing mode is intentionally intrusive and disables some normal fused paths, so the absolute throughput is not a leaderboard metric. The relative counts are still useful.

## Zero-Trailing-Split Probe

Command shape:

```text
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3,0
-dev SYCL0/SYCL1/SYCL2/SYCL3
-sm tensor -ts 1/1/1/0
```

Result:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quad2130-ts1110-p0n128-r2-20260506T153020Z.jsonl`
- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-quad2130-ts1110-p0n128-r2-20260506T153020Z.log`
- Failure: process abort at `ggml-backend.cpp:120: GGML_ASSERT(buffer) failed`

Decision: explicit zero-width trailing tensor splits are currently a llama.cpp/SYCL bug. This was a process abort, not a machine crash.

## Skip-Last Threshold Patch

Patch artifact:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-sycl-split-skip-last-below-rows-focused-20260506.patch`

The patch adds env-gated row-split logic:

```text
GGML_SYCL_SPLIT_SKIP_LAST_BELOW_ROWS=<row_threshold>
```

When set, tensors with fewer rows than the threshold are split across the first `N-1` devices and assigned zero rows on the last device. Allocation and execution both call `get_row_split()` so the row ranges match.

Validation:

- Rebuilt `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31`
- Patched default 3x sanity: `44.778557 tok/s`
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-patched-sanity-triple213-p0n128-r2-20260506T154638Z.jsonl`

Threshold sweep:

- TSV: `/home/steve/bench-results/qwen36-q4_0-gguf/skiplast-threshold-quad-assist005-p0n128-r2-20260506T154045Z.tsv`
- Env off, patched default: `37.880847 tok/s`
- `GGML_SYCL_SPLIT_SKIP_LAST_BELOW_ROWS=6144`: `38.153344 tok/s`
- `8192`: `37.894278 tok/s`
- `12288`: `37.572031 tok/s`
- `16384`: `37.819148 tok/s`

Decision: the patch is safe when unset and useful as a diagnostic control, but it should stay unset for production. It does not beat the validated `39.204149 tok/s` four-card assist run or the best `46.194319 tok/s` three-card run.

## Next

The simple threshold approach is too blunt. The next viable Q4_0 four-card direction is not "use less of the fourth card" but "avoid creating tiny independent per-token MMVQ launches for it." Candidate source work:

- graph-level decomposition where the fourth B70 handles only large independent projections;
- output-projection plus allreduce/residual epilogue fusion;
- a true four-way Q4_0 MMVQ kernel shape with lower launch and quantization overhead for narrow shards;
- using the fourth B70 for a draft/speculative model or a second session while 3x remains the main Q4_0 token path.
