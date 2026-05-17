# MiniMax Logits Gather Follow-Up - 2026-05-17

Goal: keep improving MiniMax M2.7 AutoRound W4A16 on 4x B70 while preserving
the stricter local-argmax quality bar. The current promoted strict baseline is
`cmp940h1703tpo401scj5tftf`: 60.497 output tok/s and 80.663 total tok/s at
p512/n1536 with `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`.

## Chunked Full-Logits Gather Probe

Patch surface:

- `VLLM_XPU_LOGITS_CHUNKED_GATHER=<tokens>`
- Tested chunk size: `8192`
- Required correctness flags kept enabled:
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`

Results:

| Gate | Result | Artifact |
| --- | --- | --- |
| arithmetic repeat n64/r8, no Q/K restore | fail, all token `0`/NUL | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-chunked-logits8192-20260517T020801Z.json` |
| arithmetic repeat n64/r8, Q/K restore | pass | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-qkrestore-chunked-logits8192-20260517T021357Z.json` |
| raw145 n64 exact, Q/K restore | pass, expected hash matched | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-qkrestore-chunked-logits8192-20260517T022631Z.json` |
| raw145 n96 exact, Q/K restore | fail, first NUL at token index 92 | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n96-piecewise-qkrestore-chunked-logits8192-20260517T023155Z.json` |
| raw145 n256 exact, Q/K restore | fail, 164 NUL tokens | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n256-piecewise-qkrestore-chunked-logits8192-20260517T021955Z.json` |

Interpretation: chunking the full-vocab logits all-gather is not a quality-safe
fix. It is exact through n64, then deterministically collapses to token `0`
starting at generated token index 92. Do not submit or promote speed results
from this path.

## Local-Argmax Timing

Added opt-in timing around the strict greedy local-argmax path:

- `logits.local_argmax_local_max`
- `logits.local_argmax_pair_stack`
- `logits.local_argmax_pair_all_gather`
- `logits.local_argmax_reduce`

Synchronized timing run, p512/n256:

- Log: `/home/steve/bench-results/minimax-m2.7-local-argmax-timing/vllm-minimax-m27-autoround-tp4-p512n256-20260517T023826Z.log`
- JSON: `/home/steve/bench-results/minimax-m2.7-local-argmax-timing/vllm-minimax-m27-autoround-tp4-p512n256-20260517T023826Z.json`
- Result: 18.31 output tok/s, diagnostic only because `VLLM_XPU_DECODE_TIMING_SYNC=1` adds per-region synchronization.
- Rank 0 timing after skipping first 16 tokens:
  - pair all-gather: 0.314 ms/token average
  - reduce: 0.138 ms/token average
  - pair stack: 0.118 ms/token average
  - local max: 0.088 ms/token average

Interpretation: the small pair all-gather is measurable but does not by itself
explain the full gap between the strict 60.5 tok/s result and older, less strict
65-73 tok/s graph results. The remaining speed work should focus on graph
replay/framework overhead, TTFT/prefill, and restoring a quality-safe full-logits
or fused greedy-decode path.

## Two-All-Reduce Local-Argmax Probe

Patch surface:

- `VLLM_XPU_LOCAL_ARGMAX_ALLREDUCE=1`
- First all-reduce computes the global max logit.
- Second all-reduce selects the winning global token index.

Result:

- Attempted raw145 n64 exact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-allreduce-20260517T025111Z.log`
- The run hung during warmup/profiling after AOT compile. The workers were
  orphaned and had to be killed.

Interpretation: this reducer is not usable in the current XPU graph path. Keep
it default-off and treat it as a negative diagnostic until the collective
ordering issue is understood.

## CCL Topology-Assumption Probe

Patch surface:

- `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`
- Kept strict greedy path enabled:
  - `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`

Result:

- Attempted raw145 n64 exact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-topoassume-20260517T030828Z.log`
- The engine loaded weights, loaded AOT graph artifacts, completed graph capture,
  rendered the prompt, then made no decode progress. It emitted repeated
  `No available shared memory broadcast block found in 60 seconds` warnings and
  had to be killed.

Interpretation: this CCL topology shortcut is unsafe in the current graph +
communication configuration. Do not use it for quality or speed submissions.

## Expert Parallelism Probe

Patch surface:

- Added `--enable-expert-parallel` to
  `scripts/run-vllm-minimax-quality-check.py`.
- Runtime flags:
  - `--enable-expert-parallel`
  - `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1`
  - `VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2`
  - `USE_LLM_SCALER_MOE=1`

Result:

- Attempted raw145 n64 exact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-ep4-20260517T031609Z.json`
- vLLM correctly created `TP*_EP*` ranks with 64 local experts per card.
- It fell back to the default MoE config because no tuned file exists for
  `E=64,N=1536,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json`.
- The quality gate failed: first token matched, then token `0` repeated for the
  remaining 63 generated tokens.
- Control run without llm-scaler also failed the same way:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n16-piecewise-local-argmax-ep4-no-llmscaler-20260517T032653Z.json`
  produced one correct first token followed by 15 token `0` values.

Interpretation: EP4 is not quality-safe in the current MiniMax AutoRound +
XPU path. The no-llm-scaler control means this is not only a custom INT4 MoE
kernel issue. It may need an EP-specific W4A16 MoE config and/or deeper
all-to-all/EP weight-layout debugging before it is a candidate again.

## Direct Pair-Gather Probe

Patch surface:

- `VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER=1`
- Replaces vLLM's generic `tensor_model_parallel_all_gather(local_pair,
  dim=-1)` call in the greedy local-argmax path with a direct
  `torch.distributed.all_gather_into_tensor` over the TP device group.

Quality gates:

| Gate | Result | Artifact |
| --- | --- | --- |
| raw145 n64 exact | pass, expected hash matched, no NUL/control output | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-directgather-20260517T033407Z.json` |
| raw145 n256 exact | pass, expected hash matched, no NUL/control output | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n256-piecewise-local-argmax-directgather-20260517T034002Z.json` |

Speed screen, p512/n1536:

| Run | Total tok/s | Output tok/s | Artifact |
| --- | ---: | ---: | --- |
| direct gather repeat 1 | 80.373 | 60.280 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T034250Z.json` |
| direct gather repeat 2 | 80.907 | 60.681 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T034554Z.json` |

Mean: 60.480 output tok/s and 80.640 total tok/s. This is quality-safe but
performance-neutral/slightly below the promoted 60.497 output tok/s baseline,
so it was not submitted to LocalMaxxing.

Interpretation: the generic pair all-gather wrapper is not the main bottleneck
at batch=1. A deeper fused XPU top-token reducer could still help, but it needs
to remove more framework/launch overhead than this direct gather change removes.

## No-Chunked-Prefill Probe

Patch surface:

- `--no-enable-chunked-prefill`
- Direct pair gather kept enabled for the probe.

Results:

- A quality gate with `max_num_batched_tokens=512` was rejected because vLLM
  requires `max_num_batched_tokens >= max_model_len` when chunked prefill is
  disabled.
- A quality gate with `max_num_batched_tokens=2048` passed raw145 n64 exact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-directgather-nochunkprefill-mbt2048-20260517T035004Z.json`
- The same log warns that MiniMax does not officially support disabling chunked
  prefill and then records repeated Intel compiler failures:
  `ocloc failed with error code 245`, `IGC: Internal Compiler Error: Floating
  point exception`, and `Build failed with error code: -11`.
- The p512/n1536 benchmark did not emit JSON. It entered repeated shared-memory
  broadcast waits after graph capture and had to be killed:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T035551Z.log`

Interpretation: disabling chunked prefill is not a safe MiniMax path on this
stack. It can pass a short gate but is unsupported, compiler-fragile, and
unstable under the actual benchmark shape.

## Capture-Size Narrowing Probe

Patch surface:

- `cudagraph_capture_sizes=[1]`
- `compile_sizes=[1]`
- Direct pair gather kept enabled.

Result:

- p512/n1536 completed cleanly:
  `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T040211Z.json`
- Speed regressed to 45.181 output tok/s and 60.242 total tok/s.
- The run spent 159.97 s in torch.compile and only captured one graph size,
  then decoded much slower than the strict baseline.

Interpretation: reducing capture sizes to `[1]` is not a decode improvement for
this workload. The current broader piecewise graph capture remains the better
promoted path.

## Current Decision

- Keep `VLLM_XPU_LOCAL_ARGMAX_DECODE=1` plus pair all-gather as the current
  strict quality baseline.
- Do not use `VLLM_XPU_LOGITS_CHUNKED_GATHER` for MiniMax promotion.
- Do not use `VLLM_XPU_LOCAL_ARGMAX_ALLREDUCE` in graph mode.
- Do not use `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` in the graph path.
- Do not use `--enable-expert-parallel` for MiniMax AutoRound promotion yet.
- Do not use `--no-enable-chunked-prefill` for MiniMax AutoRound on this stack.
- Do not narrow `cudagraph_capture_sizes` to `[1]` for the p512/n1536 benchmark.
- `VLLM_XPU_LOCAL_ARGMAX_DIRECT_GATHER=1` is valid but not faster than the
  current baseline; keep it as a diagnostic, not a promotion.
- Next safe optimization targets:
  - reduce TTFT/prefill without changing decode outputs;
  - tune or repair EP-specific INT4 MoE handling before retrying EP speed runs;
  - investigate a fused/local greedy-decode kernel that keeps pair selection on
    GPU without reintroducing the corrupted full-vocab logits gather.

## Later 2026-05-17 Follow-Ups

### No-Clone Allreduce With Local Argmax

Patch surface:

- `VLLM_XPU_LOCAL_ARGMAX_DECODE=1`
- `VLLM_XPU_COMPILE_ALLREDUCE_NO_CLONE=1`
- vLLM async scheduling disabled for the quality run

Result:

- Raw145 n64 exact and n256 exact had passed in separate smaller probes, but
  the extended six-prompt suite did not complete:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-extended-sixpack-n64-r2-piecewise-local-argmax-noclone-asyncoff-20260517T043730Z.log`
- The engine reached AOT load, then emitted repeated
  `No available shared memory broadcast block found in 60 seconds` warnings
  before being killed.

Interpretation: combining the strict local-argmax sampler bypass with the
no-clone compiled allreduce path is not reliable enough to publish. The older
no-clone async-off result remains a useful historical datapoint, but the current
strict greedy path should not enable no-clone until the shared-memory broadcast
stall is understood.

### Packed Single-Allreduce Local Argmax

Patch surface:

- `VLLM_XPU_LOCAL_ARGMAX_PACKED_ALLREDUCE=1`
- Added a default-off reducer that packs the local max float32 bit-order key
  and global token id into one signed int64, then calls one XCCL MAX allreduce.
- Patch snapshot:
  `patches/vllm-minimax-local-argmax-packed-allreduce-negative-20260517.patch`

Result:

- Attempted raw145 n64 exact:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-packed-allreduce-20260517T044543Z.log`
- The run did not reach generation JSON. It stalled after AOT load with the
  same repeated shared-memory broadcast warning and had to be killed.

Interpretation: the current XPU graph path does not tolerate adding a logits
stage allreduce, even when it is only one packed int64 reduction. Keep the patch
default-off as a negative diagnostic. The quality-safe local-argmax path remains
the pair all-gather reducer.

Default-off sanity after installing the packed reducer patch:

- Raw145 n64 exact passed with `VLLM_XPU_LOCAL_ARGMAX_PACKED_ALLREDUCE` unset:
  `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-defaultoff-sanity-20260517T045205Z.json`
- Combined token hash matched
  `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd`;
  `0` NUL tokens and `0` non-space control characters.

### Harness Repair

`scripts/run-minimax-strict-quality-gated-candidate.sh` was updated because it
still passed an obsolete `--compilation-config-json` argument to
`run-vllm-minimax-quality-check.py`. The wrapper now relies on the quality
script's current default piecewise graph config for the quality gates and
records the local-argmax env flags in its summary JSON.
