# 2026-05-06 Q4_0 Speculative Draft Follow-Ups

## Summary

Tested llama.cpp speculative paths after the four-card Q4_0 investigation. This did not produce a usable speed path.

The main result: a small Qwen3.5 4B GGUF draft model is now local, but it is not a useful draft for Qwen3.6 27B in the current llama.cpp/SYCL setup. CPU draft accepted no tokens, and placing the draft on the fourth B70 is blocked by the same global-device/split-buffer behavior that makes trailing or tiny fourth shards fragile.

## Downloaded Draft Candidate

- Hugging Face repo: `llmware/qwen-3.5-4b-gguf`
- File: `Qwen3.5-4B-Q4_K_M.gguf`
- Local path: `/home/steve/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`
- Size on disk: `2.6G`

The target and draft both report `tokenizer.ggml.pre = qwen35` and `n_vocab = 248320`, so the tokenizer family is compatible enough for llama.cpp to run the draft path. Acceptance was still zero in the tested prompt, which makes it a poor draft candidate.

## Draft On Fourth B70

Command shape:

```text
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3,0
target: -dev SYCL0,SYCL1,SYCL2 -sm tensor -ts 1,1,1
draft:  --spec-draft-device SYCL3 --spec-draft-ngl 99
```

Logs:

- Initial `n=64`: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-qwen35-4b-draft-target3x-draft1x-n64-20260506T160617Z.log`
- Isolated `--fit off`, `n=1`: `/home/steve/bench-results/qwen36-q4_0-gguf/spec-qwen35-4b-draftgpu-fitoff-target3x-draft1x-n1-20260506T161441Z.log`

Result:

- Both runs failed before useful generation.
- The `--fit off` run failed with Level Zero `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` during SYCL compute.

Interpretation:

The process must expose all four B70s for the draft model to use `SYCL3`, but the target tensor-split path still has code that relies on global `ggml_sycl_info().device_count`. This makes "target on first three visible devices, draft on fourth visible device" unsafe in the current source tree. Fixing this likely requires split-buffer contexts and row-split helpers to track the backend-selected device count or explicit active device list, not only the global SYCL device count.

## CPU Draft Control

Command shape:

```text
ONEAPI_DEVICE_SELECTOR=level_zero:2,1,3
target: -dev SYCL0,SYCL1,SYCL2 -sm tensor -ts 1,1,1
draft:  --spec-draft-ngl 0
```

Log:

- `/home/steve/bench-results/qwen36-q4_0-gguf/spec-qwen35-4b-draftcpu-target3x-n32-20260506T160741Z.log`

Result:

- Process returned success, but the run is not valid as a speed path.
- `n_draft = 4`
- `n_drafted = 128`
- `n_accept = 0`
- Acceptance: `0.000%`
- Target/draft decode section: `33` decoded tokens in `3.747 s`, `8.807 tok/s`
- The log contains repeated `inconsistent sequence positions` decode failures.

Decision:

The Qwen3.5 4B Q4_K_M draft should not be used for Qwen3.6 27B speculative speed claims.

## Target-Only N-Gram Attempt

Tried llama.cpp target-only n-gram speculative mode through `llama-cli` because `llama-speculative` requires `--model-draft`.

Log:

- `/home/steve/bench-results/qwen36-q4_0-gguf/cli-spec-ngrammod-target3x-n64-20260506T160921Z.log`

Result:

- Timed out after `180 s` with return code `124`.
- The log only reached memory breakdown; no useful timings were produced.

There is also a harness safety note: an earlier `llama-cli` smoke wrote hundreds of MB to stdout despite a tiny token target. Future CLI/speculative experiments should redirect stdout to `/dev/null` or use a dedicated timing harness before any longer run.

## Health Check

After these failed speculative attempts, the normal 3x llama-bench path still completed:

- JSONL: `/home/steve/bench-results/qwen36-q4_0-gguf/post-spec-sanity-triple213-p0n16-20260506T161324Z.jsonl`
- Decode: `40.818979 tok/s` on a tiny `n=16` sanity run

This was only a runtime health check, not a benchmark to compare against full `512/512` validations.

## Next

Do not spend more time on generic Qwen3.5 4B draft speculation for Qwen3.6 27B. Better speculative paths are:

- a Qwen3.6-specific MTP/DFlash draft head that shares the target distribution;
- fixing llama.cpp/SYCL split-buffer active-device accounting so target 3x plus draft 1x is possible in one process;
- continuing vLLM FP8 n-gram/MTP work where speculative infrastructure already produced real gains;
- using the fourth B70 for a second Q4 session until single-session target/draft placement is source-correct.
