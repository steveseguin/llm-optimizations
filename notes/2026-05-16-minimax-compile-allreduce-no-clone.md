# MiniMax Compile All-Reduce No-Clone Candidate

## Goal

Remove a graph-side allocation/copy target from the compiled XPU all-reduce
path without changing model math, quantization, routing, speculative decoding,
or power settings.

## Patch

Added an opt-in vLLM XPU communicator flag:

```bash
VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1
```

When torch compile is active, the XPU communicator now uses the input temporary
as the all-reduce output instead of making `input_.clone()` first. The oneCCL /
c10d all-reduce, wait, and copy-back semantics remain in place.

Patch artifact:

```text
patches/vllm-xpu-compile-allreduce-no-clone-20260516.patch
```

## Quality Gate

Strict gate passed before benchmarking:

| Check | Result |
| --- | --- |
| raw145 n64 exact token hash | `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd` |
| raw145 n256 exact token hash | `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537` |
| semantic suite, two greedy repeats | passed |
| semantic combined token hash | `adacbf144264486ea7d378ebb6a4c0ba23951b72c4cf86251a762b07ebef5805` |

Extra quality check:

- Current-clone baseline raw145 n512 hash:
  `3872a0eeddffe0e3351dca9eaf5ca7dfc8b9d0383aeb5096e398706e1fdb3244`
- No-clone raw145 n512 run matched that exact hash.
- Candidate n512 output had `nul_token_count=0` and
  `control_nonspace_text_chars=0`.

The longer default chat-prompt suite was not used as an exact gate because the
current-clone baseline was nondeterministic across two greedy repeats. It still
produced non-degenerate text, but it cannot serve as a strict equivalence
criterion.

## Benchmark

Runtime:

- `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- vLLM/XPU TP4 on 4x Intel Arc Pro B70 32GB
- `dtype=float16`
- `max_model_len=2048`
- `max_num_batched_tokens=512`
- `block_size=256`
- `--no-enable-prefix-caching`
- `--attention-backend TRITON_ATTN`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1`
- `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`
- `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1`
- `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1`
- `VLLM_XPU_USE_LLM_SCALER_MOE=1`

Results for p512/n1536, batch 1:

| Repeat | Output tok/s | Total tok/s |
| --- | ---: | ---: |
| 1 | 65.936238 | 87.914984 |
| 2 | 66.241636 | 88.322181 |
| 3 | 66.649336 | 88.865782 |
| Mean | 66.275737 | 88.367649 |

Current clean baseline mean was `65.7525` output tok/s and `87.6699` total
tok/s. The no-clone candidate improves mean decode by about `0.52 tok/s`
or `0.80%`.

LocalMaxxing accepted this result:

```text
cmp7p4cj100clo401sy8gzesk
```

## AOT Census

Baseline graph:

- 748 all-reduces
- 748 waits
- 748 copy-backs
- copy target kind: `clone` for all 748

No-clone graph:

- 748 all-reduces
- 748 waits
- 748 copy-backs
- copy target kind: `original_or_temp` for all 748

Interpretation: this patch removes the explicit clone tensor targets, but the
c10d all-reduce wait/copy-back boundary remains. That matches the small speed
increase and tells us the next meaningful communication target is eliminating or
fusing the remaining wait/copy boundary, not only the clone allocation.

## Decision

Accept as a quality-cleared small speed win and keep it behind
`VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1` while collecting broader long-form evals.
It is worth using as the base for the next MiniMax optimization pass because it
preserves strict exact canaries and improves repeatable decode speed slightly.
