# Qwen3.6 27B Q4_0 GGUF B70 Results - AOT/DNN and Driver Wedge

Date: 2026-05-03

Host: Ubuntu 24.04.4 LTS, AMD EPYC 9015 8-core, 16 logical CPUs, 16 GiB RAM plus swap, 2x Intel Arc Pro B70 / BMG-G31 32 GB.

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

llama.cpp worktree: `/home/steve/src/llama.cpp-q4-b70`, upstream `db44417` plus local experimental B70 Vulkan/SYCL patches.

## Single B70 AOT + oneDNN

Build dir: `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-aot-dnn`

Build notes:

- Intel BMG-G31 AOT needed `-fsycl-targets=intel_gpu_bmg_g31` rather than the previous `-Xsycl-target-backend --offload-arch=...` shape.
- `-Xs -ze-intel-greater-than-4GB-buffer-required` had to be skipped for Intel GPU AOT because `ocloc` rejected it.
- oneDNN linked from `/opt/intel/oneapi/dnnl/2026.0/lib/libdnnl.so.3.11`.

GPU0 result:

- Raw JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-aot-dnn-single-graph-enabled-fa0-ub64-warmup-reps3-db44417-20260503T214742Z.jsonl`
- Shape: `-dev SYCL0 -ngl 99 -p 0 -n 512 -sm none -b 512 -ub 64 -ctk f16 -ctv f16 -t 8 -fa 0 -r 3`, graph enabled, oneDNN enabled, warmup enabled.
- Result: `24.570 tok/s`, samples `24.6222`, `24.5647`, `24.5232`.

GPU1 result:

- Raw JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-aot-dnn-single-gpu1-graph-enabled-fa0-ub64-warmup-reps3-db44417-20260503T214938Z.jsonl`
- Same shape on `SYCL1`.
- Result: `24.462 tok/s`, samples `24.4926`, `24.4317`, `24.4607`.

Conclusion: AOT plus oneDNN is valid but essentially tied with the non-AOT oneDNN result (`24.553 tok/s`). It does not close the Linux gap to the Windows `27+ tok/s` Q4_0 comparison.

## Thread Sweep

AOT+DNN no-warmup single-rep thread sweep:

- `-t 4`: `24.1239 tok/s`.
- `-t 6`: `24.0468 tok/s`.
- `-t 8`: `24.0015 tok/s`.
- `-t 12`: `23.9863 tok/s`.
- `-t 16`: `23.9797 tok/s`.

Conclusion: CPU launch-thread count is not the limiting knob for this decode path.

## SYCL Debug Findings

Single-card one-token debug:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-single-n1-aot-dnn-db44417-20260503T215707Z.log`
- One decode step had `529` SYCL matmul calls.
- Q4_0 dispatch used `344` `reorder_mul_mat_vec_q4_0_q8_1_sycl` calls and `0` plain `mul_mat_vec_q4_0_q8_1_sycl` calls.

Conclusion: single-card Q4_0 is already using the reordered MMVQ kernel path. The remaining single-card gap is likely kernel/runtime efficiency rather than missing reorder.

Dual tensor one-token debug:

- Log: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-debug-dual-tensor-n1-aot-dnn-dnndis-db44417-20260503T215818Z.log`
- Shape: `-dev SYCL0,SYCL1 -sm tensor -ts 1/1 -fa 1`, one-token debug, DNN disabled.
- Failed before JSONL with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`.
- Before failure it logged `497` SYCL matmul calls, `344` reordered Q4_0 MMVQ calls, `113` explicit SYCL copy calls, and `48` memcpy-path copies for one token.

Conclusion: tensor split keeps the Q4_0 reordered MMVQ path, but copy/sync overhead is large enough to explain why dual tensor split is slower than single-card.

## Failed Forced-DMMV Experiment

- Env: `GGML_SYCL_PRIORITIZE_DMMV=1` on the combined AOT+DNN build.
- Raw JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-aot-dnn-prioritize-dmmv-single-fa0-ub64-no-warmup-reps1-db44417-20260503T215916Z.jsonl`
- Result: segmentation fault; JSONL empty.

Conclusion: do not use `GGML_SYCL_PRIORITIZE_DMMV=1` as a performance knob until isolated.

## Experimental Split Safety Rebuild

After rebuilding `/home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31-dnn` with the newer experimental split safety edits:

- Row split smoke, `-dev SYCL0,SYCL1 -sm row -ts 1/1 -fa 0 -ub 64 -n 16`, timed out after 180 seconds and wrote an empty JSONL.
- Tensor split smoke, `-dev SYCL0,SYCL1 -sm tensor -ts 1/1 -fa 1 -ub 64 -n 128`, failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` and wrote an empty JSONL.
- Tensor split one-token debug failed at the final output projection: `MUL_MAT`, `dst='result_output'`, `src0='output.weight'`, `type=q6_K`, `ne=[5120, 248320, 1, 1]`, `src1='result_norm'`.

Conclusion: the split safety edits are experimental and not validated. They address real row-split pointer hazards, but row split remains non-viable and tensor/meta split still needs separate memory/copy scheduling work.

## Driver Wedge

After the forced-DMMV segfault and follow-up split experiments, fresh single-card sanity runs started failing with Level Zero OOM in `MUL_MAT`, including the previously working AOT+DNN binary.

Failed sanity outputs:

- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-dnn-rebuild-single-sanity-fa0-ub64-n128-db44417-20260503T235314Z.jsonl` empty.
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-dnn-rebuild-single-sanity-dnndis-fa0-ub64-n128-db44417-20260503T235413Z.jsonl` empty.
- `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-aot-dnn-post-crash-single-sanity-fa0-ub64-n128-db44417-20260503T235510Z.jsonl` empty.

Kernel evidence:

- `dmesg` contains repeated Xe engine resets/coredumps on the B70s.
- `/sys/class/drm/card3/device/devcoredump` was present after the crash sequence.
- PCI reset via `/sys/bus/pci/devices/{0000:e3:00.0,0000:83:00.0}/reset` completed, but the next `sycl-ls` hung.
- `dmesg` then reported Xe TLB invalidation timeout/runtime suspend errors.
- `sycl-ls` and an attempted `pkill` are stuck in kernel `D` state.

Conclusion: the host GPU runtime should be considered wedged until reboot or equivalent driver recovery. Do not run more B70 SYCL benchmarks before recovery.
