# Qwen3.6 27B Q4_0 GGUF Four-B70 SYCL Results

Date: 2026-05-03 EDT / 2026-05-04 UTC

Host: Ubuntu 24.04.4 LTS, AMD EPYC 9015 8-core, 16 logical CPUs, 16 GiB RAM plus swap, 4x Intel Arc Pro B70 / BMG-G31 32 GB.

Model: `/home/steve/models/qwen3.6-27b-q4_0-gguf/Qwen3.6-27B-Q4_0.gguf`

llama.cpp worktree: `/home/steve/src/llama.cpp-q4-b70`, upstream `db44417` plus local experimental B70 Vulkan/SYCL patches.

## Topology

After reboot, all four B70s enumerate cleanly through Level Zero.

Level Zero order from `sycl-ls --verbose`:

- selector `0`: PCI `03:00.0`
- selector `1`: PCI `83:00.0`
- selector `2`: PCI `a3:00.0`
- selector `3`: PCI `e3:00.0`

DRM render nodes: `/dev/dri/renderD128` through `/dev/dri/renderD131`.

`steve` is in the `render` group.

Important command syntax: llama.cpp multi-GPU runs must use slash-separated devices, for example `-dev SYCL0/SYCL1`. Comma-separated devices run separate benchmark cases and are not one multi-GPU run.

## Single B70 Baseline

Clean post-reboot single-card result with only one GPU exposed:

- Raw JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-four-b70-single-selector0-fa0-ub64-warmup-reps3-db44417-20260504T004035Z.jsonl`
- Shape: `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `-dev SYCL0`, `-sm none`, `-fa 0`, `-ub 64`, graph enabled, oneDNN enabled, AOT+DNN build.
- Result: `24.723 tok/s`, samples `24.7981`, `24.7432`, `24.6271`.

All four isolated Level Zero selectors passed a 16-token smoke at about `24.67-24.73 tok/s`.

## Dual B70

Layer split, selector `1,3`:

- Shape: `ONEAPI_DEVICE_SELECTOR=level_zero:1,3`, `-dev SYCL0/SYCL1`, `-sm layer -ts 1/1`, `-fa 0`, 512 tokens, 3 reps.
- Result: `24.152 tok/s`, samples `24.1135`, `24.1699`, `24.1735`.
- Interpretation: stable but no single-session speedup.

Tensor split, selector `1,3`:

- Shape: `ONEAPI_DEVICE_SELECTOR=level_zero:1,3`, `-dev SYCL0/SYCL1`, `-sm tensor -ts 1/1`, `-fa 1`, 512 tokens, 3 reps.
- Result: `26.461 tok/s`, samples `26.6698`, `26.6279`, `26.0856`.

Pair sweep at 128 tokens, tensor split, equal `1/1`:

- Pair `0,1`: `26.661 tok/s`
- Pair `0,2`: `26.652 tok/s`
- Pair `0,3`: `26.682 tok/s`
- Pair `1,2`: `26.624 tok/s`
- Pair `1,3`: `26.430 tok/s`
- Pair `2,3`: `26.213 tok/s`

Best validated dual tensor run:

- Raw JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/sycl-four-b70-dual-tensor-pair03-512-reps3-db44417-20260504T010516Z.jsonl`
- Shape: `ONEAPI_DEVICE_SELECTOR=level_zero:0,3`, `-dev SYCL0/SYCL1`, `-sm tensor -ts 1/1`, `-fa 1`, `-ub 64`, graph enabled, oneDNN disabled, non-AOT DNN-off build.
- Result: `26.872 tok/s`, samples `26.892`, `26.856`, `26.8685`.
- Improvement over clean single-card: about `+8.7%`.
- Quality status: quality-preserving Q4_0 GGUF, f16 KV, no speculative decoding, no quantization change.

LocalMaxxing submission:

- ID: `cmoqinkvt000fjv04ps77c42h`
- Submitted under base model `z-lab/Qwen3.6-27B-DFlash`, quantization `Q4_0`.
- GGUF distribution repo noted as `spiritbuun/Qwen3.6-27B-DFlash-GGUF`.
- API caveat: LocalMaxxing rejected `backend=sycl-level-zero`, so backend was omitted and SYCL/Level Zero was recorded in notes/extra flags.
- Parser caveat: the API parsed llama.cpp `-t 8` as sampler temperature. Future submissions should provide explicit sampler fields.

## Four B70

Layer split, selector `0,1,2,3`:

- Shape: `-dev SYCL0/SYCL1/SYCL2/SYCL3`, `-sm layer -ts 1/1/1/1`, `-fa 0`, 128-token smoke.
- Result: `23.340 tok/s`.
- Interpretation: worse than single/dual; layer split remains a capacity path, not a decode-speed path.

Tensor split, selector `0,1,2,3`:

- Shape: `-dev SYCL0/SYCL1/SYCL2/SYCL3`, `-sm tensor -ts 1/1/1/1`, `-fa 1`, 32-token smoke.
- Result: `16.548 tok/s`.
- Interpretation: current 4-way tensor split has too much overhead and is not viable yet.

## Tuning Notes

Best-pair ubatch sweep at 256 tokens:

- `-ub 32`: `26.669 tok/s`
- `-ub 64`: `26.674 tok/s`
- `-ub 128`: `26.223 tok/s`
- `-ub 256`: `26.718 tok/s`
- `-ub 512`: `26.186 tok/s`

No clear win over the validated `-ub 64` 512-token result.

Build comparisons at 256 tokens:

- AOT+DNN build, oneDNN disabled: `26.233 tok/s`
- AOT+DNN build, oneDNN enabled: `26.638 tok/s`
- Non-AOT DNN-enabled build: `26.560 tok/s`
- Best remains non-AOT DNN-off build with tensor split.

Asymmetric tensor split ratios:

- `2/1`, `1/2`, `3/2`, and `2/3` abort at `ggml-backend-meta.cpp:1014`.
- `3/1`: `25.150 tok/s`.
- `1/3`: `21.946 tok/s`.
- Keep `-ts 1/1` until the meta-scheduler assertion is fixed.

## Next Work

- Push the dual tensor path over `27 tok/s` first; it is close.
- Profile or instrument tensor split copy/sync overhead; current dual tensor is only `+8.7%`, not the desired `80%+`.
- Investigate why 4-way tensor split collapses to `16.5 tok/s`.
- Fix or understand `ggml-backend-meta.cpp:1014` assertions for asymmetric tensor splits.
- Keep single-card runs isolated with `ONEAPI_DEVICE_SELECTOR=level_zero:N`.
