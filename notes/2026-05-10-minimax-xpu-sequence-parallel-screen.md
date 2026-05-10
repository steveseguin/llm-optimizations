# MiniMax M2.7 XPU Sequence-Parallel Screen, 2026-05-10

## Purpose

Test whether vLLM's upstream sequence-parallel communication-boundary pass can
help the 4x B70 MiniMax M2.7 AutoRound INT4 path. This was a direct response to
the current bottleneck theory: too much latency is being paid around TP
collectives, residual/RMSNorm boundaries, Q/K normalization, and graph
scheduling.

The target model and clean recipe remain:

- model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM XPU, TP4, FP16 activations
- MoE fast path: `VLLM_XPU_USE_LLM_SCALER_MOE=1`
- quality guardrails: no expert dropping, no speculative decode, no skipped
  Q/K TP variance allreduce, no power-limit changes

## Upstream Context

vLLM documents sequence parallelism as a prerequisite for AsyncTP GEMM and
collective overlap. Its intended transform is:

`AllReduce -> RMSNorm` becomes `ReduceScatter -> local RMSNorm -> AllGather`.

The same vLLM documentation says this path is only tested on NVIDIA CUDA, may
work on ROCm, and requires explicit `sp_min_token_num` outside the current
autoconfigured SM90 path:

- https://docs.vllm.ai/en/stable/design/fusions/

Intel llm-scaler remains relevant as an XPU-specific reference because its local
README targets Arc Pro B60/B70, CCL P2P/USM, INT4/FP8 serving, and distributed
parallel modes:

- `/home/steve/src/llm-scaler/README.md`
- `/home/steve/src/llm-scaler/vllm/README.md`

## Local Patch

Patch artifact:

- `patches/vllm-xpu-sequence-parallel-screen-20260510.patch`

The patch is default-off. It does three things:

1. Allows XPU to keep `enable_sp` when `VLLM_XPU_EXPERIMENTAL_ENABLE_SP=1`.
2. Imports `SequenceParallelismPass` on XPU when explicitly enabled.
3. Skips static-FP8 SP pattern registration on XPU, because the MiniMax
   AutoRound path is W4A16 and the FP8 pattern construction hit a TorchInductor
   `torchbind` converter failure before any benchmark could run.

The already-existing custom collective switch,
`VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1`, was required for the SP pass to compile
far enough to run. Without it, the pattern matcher captures the TP process-group
object and fails during pattern conversion.

## Runs

| Run | Status | Total tok/s | Output tok/s | Finding |
| --- | --- | ---: | ---: | --- |
| SP opt-in, default allreduce | failed | none | none | `SequenceParallelismPass` was not imported on XPU. |
| SP opt-in after import patch | failed | none | none | static-FP8 SP pattern hit TorchInductor `torchbind` conversion failure. |
| SP opt-in, FP8 patterns skipped | failed | none | none | normal RMS SP pattern still hit `torchbind` conversion failure because allreduce captured the TP group object. |
| SP opt-in plus custom-op collectives | completed negative | `17.196906` | `5.732302` | Compiled and ran, but much slower than the clean p64/n32 reference. |

Completed negative run:

- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-sp-customar-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T193942Z.log`
- json: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-sp-customar-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T193942Z.json`
- command shape: p64/n32, TP4, `max_model_len=512`,
  `max_num_batched_tokens=256`, `sp_min_token_num=1`
- compile time: `103.06 s`
- elapsed benchmark time: `5.582399606 s`
- total tokens: `96`
- output tokens: `32`

Reference comparison:

- clean p64/n32 health check after llm-scaler ESIMD core quarantine:
  `77.330536` total tok/s and `25.776845` output tok/s
- log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-smoke-after-esimd-quarantine/vllm-minimax-m27-autoround-tp4-p64n32-20260510T185306Z.log`
- post-SP-patch env-off clean smoke: `76.267703` total tok/s and
  `25.422568` output tok/s
- post-SP-patch log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-post-sp-clean-smoke-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T194412Z.log`

## Conclusion

Do not promote upstream SP on B70 today. The path is mathematically attractive
but operationally wrong for this stack:

- XPU must bypass platform disables and import CUDA-oriented pass code.
- FP8 static patterns need to be disabled for the W4A16 MiniMax path.
- The normal allreduce pattern cannot be converted unless allreduce is routed
  through the custom-op collective wrapper.
- Once it runs, throughput is roughly `4.5x` slower than the clean p64/n32
  smoke reference.

This does not invalidate the communication-boundary direction. It means the
next useful work is XPU-native fusion around the specific MiniMax boundaries,
not forcing the generic CUDA/Hopper SP pass. Prioritize:

1. C++/SYCL or compiler-pass fusion for hidden-state allreduce plus residual
   add/RMSNorm.
2. Q/K RMS variance reduction fused with adjacent qkv/RoPE work.
3. MoE output epilogue plus allreduce plus residual/RMSNorm fusion.

No LocalMaxxing submission: valid negative engineering result, but not a
leaderboard-worthy datapoint.
