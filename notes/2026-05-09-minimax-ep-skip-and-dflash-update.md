# 2026-05-09 MiniMax EP Skip and DFlash Update

## Expert-Parallel Skip Test

Goal: keep non-local expert ids as `-1` in the vLLM EP expert map and make the
llm-scaler tiny u4 decode kernels skip those rows instead of remapping every
non-local expert to local expert `0`.

This tests whether EP was losing most of its time doing useless local GEMV work
for experts that belong to another rank.

| run | prompt/output | total tok/s | output tok/s convention | result |
| --- | ---: | ---: | ---: | --- |
| EP Python map before kernel skip | 1/8 | 16.795602 | 14.929423 | smoke pass |
| EP kernel skip for `expert < 0` | 1/8 | 16.883004 | 15.007116 | smoke pass, only +0.5% |
| stable non-EP BF16 u4 decode | 512/512 | 73.215399 | 36.607699 | current best quality-preserving AutoRound path |

The skip path is functional, but the gain is too small to promote. EP remains
far slower than the non-EP custom u4 decode path, which means the remaining EP
penalty is likely communication, scheduler, or all-to-all overhead rather than
wasted non-local expert GEMVs.

Relevant patches:

```text
patches/vllm-minimax-ep-u4-expert-map-skip-20260509.patch
patches/llm-scaler-minimax-ep-u4-skip-20260509.patch
```

Relevant logs:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260509T220433Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260509T221627Z.log
```

## DFlash Speculative Decode

I retested the MiniMax DFlash drafter with `num_speculative_tokens=3` rather
than 4. The reason is specific to our current bridge: vLLM verification likely
uses `num_speculative_tokens + 1` rows, and 3 keeps the verifier at 4 rows,
which still hits the decode-only llm-scaler u4 path.

DFlash remains negative on this stack.

| run | result | key detail |
| --- | --- | --- |
| p1/n32, default GPU utilization, async | no KV memory | available KV cache was `-0.61 GiB` |
| p1/n32, GPU util 0.98, async | startup reservation failure | requested reservation exceeded free memory |
| p1/n32, GPU util 0.93, async | Level Zero device lost | `UR_RESULT_ERROR_DEVICE_LOST` in `WorkerAsyncOutputCopy` |
| p1/n32, GPU util 0.93, no async | no KV memory | available KV cache was `-0.29 GiB` |
| p1/n8, GPU util 0.95, no async, smaller batch | generation hang | KV cache allocated, then progress stayed at 0/1 prompts until killed |

Quality note: DFlash is target-verified speculative decoding. If it completes
correctly, accepted tokens preserve target-model output quality. These attempts
did not complete, so there is no valid throughput or quality result to submit.

Logs:

```text
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n32-20260509T222426Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n32-20260509T223235Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n32-20260509T223320Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n32-20260509T224356Z.log
/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p1n8-20260509T225220Z.log
```

## Conclusion

Do not spend more time on DFlash or EP skip until a new runtime or vLLM change
alters the conditions. N-gram speculation is also unlikely to improve the
random-prompt benchmark because the synthetic prompts have little repetition.
It may still be useful for real workloads with repeated boilerplate.

The next useful work is back on the stable non-EP AutoRound path:

- profile BF16 p512/n512 to split time between MoE, attention, and TP
  communication;
- keep router/logits fusion and DFlash disabled for benchmarks;
- reduce iteration time by moving the AutoRound model off the external NTFS
  drive or adding enough RAM for the filesystem cache.
