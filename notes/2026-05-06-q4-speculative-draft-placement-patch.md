# 2026-05-06 Q4_0 Speculative Draft Placement Patch

## Summary

Added a narrow local patch to `llama-speculative` so a draft model does not inherit the target model's tensor-parallel split mode when the draft is explicitly assigned to one device or to CPU.

This fixed the placement bug:

- target Qwen3.6 27B Q4_0 can remain on three B70s through `-sm tensor -dev SYCL0,SYCL1,SYCL2 -ts 1,1,1`;
- draft Qwen3.5 4B can be placed on `SYCL3` with `--spec-draft-device SYCL3`;
- `--spec-draft-device none` now produces CPU model, CPU KV, and CPU compute buffers instead of creating a four-device meta backend.

It did not produce a usable speculative speed path. The Qwen3.5 4B draft still hangs or times out in the common init/warmup path with repeated SYCL pool allocation failures around `889,986,048` bytes. This is not a LocalMaxxing benchmark.

## Patch

Patch artifact:

- `/home/steve/llm-optimization-artifacts/patches/llama-cpp-speculative-draft-single-device-splitmode-20260506.patch`

Source file changed locally:

- `/home/steve/src/llama.cpp-q4-b70/examples/speculative/speculative.cpp`

Behavior:

- If `params.speculative.draft.devices` is explicitly `none`, set draft load to `LLAMA_SPLIT_MODE_NONE` and `main_gpu = -1`.
- If exactly one draft device is specified and the target is using `LLAMA_SPLIT_MODE_TENSOR`, set draft load to `LLAMA_SPLIT_MODE_NONE` with that single device.
- If multiple draft devices are specified, preserve tensor split behavior.

Build:

```text
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-speculative -j2
```

## Controls

Target-only control with all four GPUs visible but target selected on three:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/targetonly-4visible-3selected-p0n16-r1-20260506T162205Z.log`
- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/targetonly-4visible-3selected-p0n16-r1-20260506T162205Z.jsonl`
- Result: `41.001202 tok/s` for tiny `n=16`
- Meaning: the fourth visible B70 alone does not break the three-card target path.

Draft standalone through `llama-bench`, all four GPUs visible, draft on `SYCL3`:

- JSONL: `/home/steve/bench-results/qwen35-4b-gguf/sycl-standalone-4visible-dev3-p0n1-r1-20260506T164115Z.jsonl`
- Log: `/home/steve/bench-results/qwen35-4b-gguf/sycl-standalone-4visible-dev3-p0n1-r1-20260506T164115Z.log`
- Result: `85.113793 tok/s` for tiny `n=1`
- Meaning: Qwen3.5 4B can run on `SYCL3` in `llama-bench`.

Draft standalone through `llama-cli`, all four GPUs visible, draft on `SYCL3`:

- Log: `/home/steve/bench-results/qwen35-4b-gguf/cli-standalone-4visible-dev3-n1-20260506T164224Z.log`
- Result: timeout after `120 s`, log file remained empty because CLI output was redirected away.
- Meaning: `llama-cli`/common-init behavior differs from `llama-bench`; do not use `llama-bench` alone to validate this draft path.

## Patched Speculative Attempts

CPU draft placement:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-cpudraft-none-4visible-target3x-n1-20260506T162839Z.log`
- Placement: target `Meta()` on three B70s; draft CPU model, CPU KV, CPU compute.
- Result: timeout after `180 s`.
- Failure signal: repeated `alloc: can't allocate 889986048 Bytes of memory on device/GPU`.

Draft on fourth B70, Q8 cache enabled:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-draftsycl3-patched-target3x-n1-20260506T163159Z.log`
- Placement: target `Meta()` on three B70s; draft model/KV/compute on `SYCL3`.
- Result: timeout after `180 s`.
- Failure signal: repeated `alloc: can't allocate 889986048 Bytes of memory on device/GPU`.

Draft on fourth B70, Q8 cache disabled:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-draftsycl3-patched-noq8-target3x-n1-20260506T163523Z.log`
- Placement: target `Meta()` on three B70s; draft model/KV/compute on `SYCL3`.
- Result: timeout after `180 s`.
- Failure signal remained, so this is not only the local Q8-cache optimization.

Draft on fourth B70, smaller context/batch, Q8 cache disabled:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-draftsycl3-patched-noq8-c512-target3x-n1-20260506T163903Z.log`
- Context/batch: `-c 512 -b 512 -ub 16`
- Placement: target `Meta()` on three B70s; draft model/KV/compute on `SYCL3`.
- Result: timeout after `120 s`.
- Failure signal remained and allocation size did not change, so this is not ordinary KV or scheduler buffer pressure.

## Conclusion

The local patch fixes a real `llama-speculative` placement problem, but generic Qwen3.5 4B is still not a useful Qwen3.6 27B draft path on this SYCL setup.

Next useful work:

- keep the patch as a reproducibility artifact;
- avoid more time on generic Qwen3.5 4B speculation unless investigating common-init/SYCL pool behavior directly;
- prefer Qwen3.6-specific MTP/DFlash or vLLM FP8 speculative paths;
- if returning to llama.cpp speculation, add a cleaner draft-specific split-mode/device model parameter instead of mutating shared `common_params`.
