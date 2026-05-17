# MiniMax Gather-Broadcast Argmax Follow-Up - 2026-05-17

Goal: keep improving MiniMax M2.7 AutoRound W4A16 on 4x Intel Arc Pro B70
without lowering the quality bar. This pass focused on the strict greedy
local-argmax decode path, where the current promoted result is 61.464 output
tok/s and 81.953 total tok/s at p512/n1536.

## Patch Surface

- Added `VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST=1` to the local-argmax path in
  `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`.
- The candidate gathers each rank's `(max_logit, global_token_id)` pair to the
  TP root, reduces on root, then broadcasts the selected token id to every TP
  rank.
- Updated `scripts/run-minimax-strict-quality-gated-candidate.sh` so candidate
  summaries record `VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST`.
- The mode is default-off and must pass exact-token gates before benchmarking.

## Timing Context

Low-overhead diagnostic run, p512/n512:

- Log: `/home/steve/bench-results/minimax-m2.7-low-overhead-timing/vllm-minimax-m27-autoround-tp4-p512n512-20260517T081945Z.log`
- JSON: `/home/steve/bench-results/minimax-m2.7-low-overhead-timing/vllm-minimax-m27-autoround-tp4-p512n512-20260517T081945Z.json`
- Throughput with timing instrumentation enabled: 29.651 output tok/s, 59.302
  total tok/s. Treat this as diagnostic only.

Rank 0 timing summary after skipping the first 64 decode tokens:

| Region | Count | Avg ms/token | Max ms |
| --- | ---: | ---: | ---: |
| `logits.local_argmax_pair_all_gather` | 448 | 7.858599 | 9.489847 |
| `logits.local_argmax_reduce` | 448 | 0.103913 | 0.114541 |
| `logits.local_argmax_pair_stack` | 448 | 0.065498 | 0.174832 |
| `logits.local_argmax_local_max` | 448 | 0.038404 | 0.048720 |

Interpretation: the tiny pair collective is the largest CPU-visible timed region
inside token selection, but timing wrappers and compile/runtime effects make the
absolute throughput non-comparable to normal benchmark runs.

## Rejected: Two-Allreduce Argmax

Patch surface:

- `VLLM_XPU_LOCAL_ARGMAX_ALLREDUCE=1`
- First allreduce computes max logit value; second allreduce selects token id.

Strict quality-only gate:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-twoallreduce-qualityonly-strict-tp4-ctx2048-mbt512-bs256-20260517T083040Z-summary.json`
- Status: `quality_failed_raw145_n64`
- Failure mode: corrupt/invalid token id `-4`, exact token hash mismatch.

Decision: do not benchmark or promote this mode. It is not quality-safe on the
current XPU/oneCCL stack.

## Accepted Diagnostic: Gather-Broadcast Argmax

Strict quality-only gate:

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-gatherbroadcast-qualityonly-strict-tp4-ctx2048-mbt512-bs256-20260517T083707Z-summary.json`
- Quality directory: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimaxlogits-localargmax-gatherbroadcast-qualityonly-strict-tp4-ctx2048-mbt512-bs256-20260517T083707Z-quality`

Quality gates:

| Gate | Result |
| --- | --- |
| raw145 n64 exact | pass, `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd` |
| raw145 n256 exact | pass, `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537` |
| semantic n64/r2 | pass, PASS/arithmetic/code checks deterministic |
| arithmetic repeat n64/r8 | pass, 8 deterministic greedy calls containing `42` |

Benchmark shape:

- Model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Runtime: vLLM 0.20.1-local, TP4, float16, XPU graph, oneCCL OFI
- Prompt/output: p512/n1536, batch 1, greedy temperature 0
- Flags: MiniMax logits MoE path, Q/K clean-weight guard, delayed attention
  allreduce, block-size 256, prefix cache off, TRITON_ATTN.

Speed results:

| Run | Output tok/s | Total tok/s | Artifact |
| --- | ---: | ---: | --- |
| repeat 1 | 61.403687 | 81.871583 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T085018Z.json` |
| repeat 2 | 61.639893 | 82.186525 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T085449Z.json` |
| repeat 3 | 61.384402 | 81.845869 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T085750Z.json` |

Mean: 61.475994 output tok/s and 81.967992 total tok/s.

LocalMaxxing:

- Submitted as `cmp9jyd8m049ko401kk2n1pju`.
- Payload: `/home/steve/llm-optimizations-publish/data/localmaxxing-minimax-m27-autoround-gatherbroadcast-localargmax-p512n1536-20260517.payload.json`
- Response: `/home/steve/llm-optimizations-publish/data/localmaxxing-responses/minimax-m27-autoround-gatherbroadcast-localargmax-p512n1536-20260517.response.json`
- The submission notes explicitly state that this is a quality-safe
  noise-level tie, not a meaningful speed breakthrough.

One additional retry failed before model load with `oneCCL:
atl_ofi_comm.cpp:232 init_transport: EXCEPTION: failed to initialize ATL`.
The follow-up recovery check showed all 4 Level Zero devices enumerating, and
the next retry completed normally. Treat this as a transient oneCCL/OFI startup
failure, not a model or quality failure.

## Decision

`VLLM_XPU_LOCAL_ARGMAX_GATHER_BROADCAST=1` is quality-safe under the current
strict gate, but its mean speed is only 0.0115 output tok/s above the previous
strict result, about +0.02%. That is a noise-level tie, not a meaningful
performance improvement.

Keep it as a diagnostic/default-off path. The next serious decode work should
not be more reshuffling of tiny pair collectives unless it removes a whole
framework/communication boundary. More promising targets:

- GPU-resident top-token selection plus next-token handoff with fewer Python or
  framework callbacks.
- A graph-safe fused sampler/top-token path that avoids full-vocab logits gather
  and avoids per-token CPU-visible collective orchestration.
- EP4 repair for MiniMax AutoRound, starting with the EP-specific W4A16 MoE
  config/layout and all-to-all path, because EP remains the largest theoretical
  upside if quality can be restored.
- TTFT/prefill tuning, but only if decode quality and decode throughput do not
  regress.