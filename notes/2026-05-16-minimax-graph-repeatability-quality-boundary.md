# MiniMax M2.7 Graph Repeatability Quality Boundary - 2026-05-16

## Context

The earlier MiniMax M2.7 AutoRound W4A16 TP4 result reached about 62.66 output tok/s, but it was only protected by raw token-hash and weaker semantic smoke checks. A stricter prompt-scoped repeatability gate exposed a correctness issue: repeated greedy calls in one persistent vLLM engine can return a bad answer on the seventh arithmetic request.

This is not a quantization-quality conclusion yet. The failures look like runtime state or graph replay drift: six repeated arithmetic calls return `42`, the seventh produces unrelated token fragments, and the eighth returns `42` again.

## Harness Changes

Updated `scripts/run-vllm-minimax-quality-check.py`:

- Added prompt-scoped semantic checks with `--require-prompt-substring INDEX:TEXT` and `--require-prompt-regex INDEX:REGEX`.
- Added repeatability modes: `token`, `text`, `lstrip_text`, `normalized_text`, and `none`.
- Added per-prompt determinism diagnostics for token/text/lstrip/normalized hashes.

Updated `scripts/run-minimax-strict-quality-gated-candidate.sh`:

- The semantic suite now checks each prompt against its own expected answer.
- Added `arithmetic-repeat-n64-r8` by default to catch persistent-engine request-state drift before any benchmark runs.
- Kept raw145 exact hash checks as backend-specific pattern canaries.

## Key Results

Fast piecewise graph path, no-clone allreduce, async scheduling off:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-no-clone-asyncoff-20260516T215434.json`
- Result: failed repeatability.
- Runs 0-5 and 7: `\n\n42`, token IDs `[367, 5130, 200020]`.
- Run 6: `\n\n Aux Auxérence Aux`, token IDs `[367, 50795, 50795, 150841, 50795, 200020]`.

Synchronous XPU sampled-token CPU copy:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-sync-to-list-20260516T223227Z.json`
- Result: failed repeatability.
- Run 6: `\n\n Generate Aux衰老 Aux`, token IDs `[367, 50818, 50795, 100814, 50795, 200020]`.
- Interpretation: the simple nonblocking sampled-token D2H copy path is not the root cause.

Piecewise graph with `cudagraph_copy_inputs=true`:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-copy-inputs-20260516T224119Z.json`
- Result: failed repeatability.
- Run 6: `\n\nత Auxత Aux`, token IDs `[367, 100825, 50795, 100825, 50795, 200020]`.
- Interpretation: copying captured graph inputs is not enough to fix the replay drift.

Piecewise graph with `compile_sizes=[]`:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-no-compile-size-20260516T224714Z.json`
- Result: failed repeatability.
- Run 6: `\n\n perubahan perubahan perubahan Fang`, token IDs `[367, 50793, 50793, 50793, 100809, 200020]`.
- Interpretation: avoiding the explicit decode compile size does not avoid the problematic captured path.

`cudagraph_mode=NONE`:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-cudagraph-none-20260516T223544Z.json`
- Result: passed arithmetic repeatability.
- All eight runs: `\n\n42`, token IDs `[367, 5130, 200020]`.
- However, full strict gate stopped on raw145 exact pattern mismatch:
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-cudagraph-none-repeatgate-strict-tp4-ctx2048-mbt512-bs256-20260516T225236Z-quality/raw145-n64-exact.json`
  - Expected graph-path pattern includes `alpha beta gamma delta ...`.
  - `cudagraph_mode=NONE` output skipped `delta` in the first cycle: `alpha beta gamma epsilon ...`.
- Interpretation: full graph disable is not currently quality-equivalent, even though it fixes the repeat arithmetic drift.

## Current Working Hypothesis

The >60 tok/s graph path is not yet promotion-safe. The quality boundary is inside XPU graph replay or captured model state, not the model weights alone and not the simplest sampled-token host copy path. Full graph disable changes numerical/output behavior on the raw145 pattern canary, so it is not an acceptable final workaround.

Most likely areas:

- XPU graph replay state for MiniMax M2.7 decode/prefill boundaries.
- Captured buffers used by MiniMax attention, Q/K RMS norm, router, or INT4 MoE path.
- Scheduler or block-table metadata captured by piecewise graphs and reused across sequential requests.
- XPU graph communication capture around TP allreduce, especially with the no-op communicator capture workaround.

## Next Tests

- `cudagraph_num_of_warmups=8`:
  - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-warmups8-20260516T225538Z.json`
  - Result: failed repeatability.
  - Run 6: `\n\n_vec Aux_vec Aux`, token IDs `[367, 50809, 50795, 50809, 50795, 200020]`.
  - Interpretation: extra graph warmup before capture does not fix the replay drift.
- Instrument graph replay inputs/outputs around the run-6 boundary.
- Try disabling only specific MiniMax fused helpers while keeping piecewise graphs:
  - llm-scaler INT4 MoE path as a control, even if slower.
    - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-no-llm-scaler-moe-20260516T230210Z.json`
    - Result: failed immediately with NUL-token collapse.
    - Interpretation: disabling this path is not a quality-safe workaround. It does not isolate the graph repeat drift by itself because the fallback path is currently broken for this model/stack.
  - Q/K RMS helper or restore path.
  - Attention delayed-allreduce patch.
    - JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-no-attn-delay-20260516T230810Z.json`
    - Result: failed repeatability.
    - Run 6: `\n\n Aux Aux Aux Aux`, token IDs `[367, 50795, 50795, 50795, 50795, 200020]`.
    - Interpretation: the drift is not fixed by disabling the delayed-attention allreduce patch.
- Add a backend-specific exact raw hash policy instead of assuming one raw145 token hash applies across graph modes.

## Additional Diagnostic Results

Piecewise graph with explicit `torch.xpu.synchronize()` before and after graph replay:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-sync-replay-20260516T231711Z.json`
- Result: failed repeatability.
- Run 6: `\n\n Generate Aux衰老 Aux`, token IDs `[367, 50818, 50795, 100814, 50795, 200020]`.
- Interpretation: the failure is not a simple missing replay stream synchronization.

Piecewise graph with strong output references instead of weak output references:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-strong-output-20260516T232444Z.json`
- Result: failed repeatability.
- Run 6: `\n\n_vec Aux_vec Aux`, token IDs `[367, 50809, 50795, 50809, 50795, 200020]`.
- Interpretation: the failure is not explained by captured output lifetime through weak references.

Piecewise graph with compiled prefill skipped:

- First attempt log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-skip-prefill-20260516T233549Z.log`
- Result: startup failure, `Expected exactly one compiled range_entry for static shape compilation, but found 2`.
- Retry JSON with `--disable-auto-compile-ranges`: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-skip-prefill-noautorange-20260516T234028Z.json`
- Retry result: failed repeatability.
- Retry run 6: `\n\n完工 Aux衰老 Aux`, token IDs `[367, 100847, 50795, 100814, 50795, 200020]`.
- Interpretation: the repeatability drift is not isolated to compiled prefill.

Piecewise graph with recapture after four replays:

- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-recapture4-20260516T234523Z.log`
- Result: runtime failure, `CUDA graph capturing detected at an inappropriate time`.
- Interpretation: recapturing inside normal decode replay is not currently operational with vLLM's graph-capture guard.

Piecewise graph with eager Q/K norm:

- Log: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-eager-qk-norm-20260516T235146Z.log`
- Result: startup failure because TorchDynamo rejects a `torch.compiler.disable()`d eager helper during compilation.
- Interpretation: this is not a valid candidate in the current compiled path.

Piecewise graph with decomposed Q/K norm:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-decomposed-qk-norm-20260516T235405Z.json`
- Result: failed repeatability.
- Run 6: `\n\n Generate Aux衰老 Aux`, token IDs `[367, 50818, 50795, 100814, 50795, 200020]`.
- Interpretation: the Q/K norm expression itself is not the primary root cause.

Piecewise graph with `FLASH_ATTN` instead of `TRITON_ATTN`:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-flash-attn-20260516T235952Z.json`
- Result: failed repeatability.
- Run 6: `\n\nత Auxత Aux`, token IDs `[367, 100825, 50795, 100825, 50795, 200020]`.
- Interpretation: the failure is not specific to the Triton attention backend.

Piecewise graph with monolithic llm-scaler MiniMax INT4 MoE logits path:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-moe-minimax-logits-20260517T000553Z.json`
- Log confirmed: `Using llm-scaler XPU INT4 MiniMax logits decode path`.
- Result: failed repeatability.
- Run 6: `\n\n_vec Aux_vec Aux`, token IDs `[367, 50809, 50795, 50809, 50795, 200020]`.
- Interpretation: separate router/expert framework handoff is not the sole cause.

Piecewise graph with logprob capture on the arithmetic repeat canary:

- JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r8-piecewise-logprobs-20260517T001202Z.json`
- Result: failed repeatability.
- Run 6: `\n\n_vec Aux_vec Aux`, token IDs `[367, 50809, 50795, 50809, 50795, 200020]`.
- Important observation: the first generated token remains normal, but the second generated token and later wrong-token steps report NaN logprobs. Good runs pick `42` with finite logprob around `-0.0008`; bad run 6 selects a rank-0 token with NaN logprob.
- Interpretation: this is hard logits or hidden-state corruption, not a close numerical tie. The >60 tok/s path remains untrusted until the first NaN source is localized and removed.

## Next Tests After NaN Evidence

- Use filtered finite tracing for high-level decode tensors:
  - `gpu_model_runner.hidden_states_before_logits_index`
  - `gpu_model_runner.logits_after_compute`
  - `logits_processor.hidden_states`
  - `logits_processor.local_logits`
  - `logits_processor.gathered_logits`
  - `logits_processor.trimmed_logits`
- If hidden states before logits are already nonfinite, trace MiniMax layer boundaries to find the first layer that produces NaNs.
- If hidden states are finite and logits become nonfinite, instrument LM-head/logits gather and vocab-parallel postprocessing.
- Do not submit additional LocalMaxxing results for this path until the arithmetic repeat canary and nonfinite gate pass.

## 2026-05-17 Update: Greedy Local-Argmax Decode Bypass

Finite tracing found the first visible corruption at the tensor-parallel full-vocab logits gather:

- Hidden states before logits were finite.
- Local LM-head logits were finite.
- `logits_processor.gathered_logits` developed NaNs and very large invalid values during the repeat drift.

For greedy `temperature=0` decode, full-vocab logits are not needed on the CPU or on every rank. The next patch adds a guarded local-argmax path:

- Each TP rank computes its local top `(logit, vocab_index)` from local vocab-parallel logits.
- Ranks gather only the small top-pair tensor instead of the full vocab logits tensor.
- The global top token is selected from those pairs and injected into the sampler path.
- The shortcut is enabled only when it preserves greedy argmax semantics: no speculative decode, no logprobs, no nonzero temperature sampling, no penalties, no bad-word or allowed-token masks, and no active non-argmax logits processors.

This is also the cleanest version of the "reduce CPU/framework callbacks" idea for this specific failure: the runtime no longer materializes and transfers the entire gathered logits surface for a single greedy token decision.

### Patch Surface

Source tree:

- `/home/steve/src/vllm/vllm/model_executor/layers/logits_processor.py`
- `/home/steve/src/vllm/vllm/model_executor/models/minimax_m2.py`
- `/home/steve/src/vllm/vllm/v1/worker/gpu_model_runner.py`
- `/home/steve/src/vllm/vllm/benchmarks/throughput.py`
- diagnostic helpers in `vllm/compilation/cuda_graph.py`, `vllm/distributed/device_communicators/base_device_communicator.py`, and `vllm/utils/xpu_finite_trace.py`

Runtime gate:

```bash
VLLM_XPU_LOCAL_ARGMAX_DECODE=1
VLLM_BENCH_TEMPERATURE=0
```

The benchmark temperature override is needed because `vllm bench throughput` defaults to `temperature=1.0`; without the override, the guarded path correctly refuses to activate.

### Quality Gates

All of the following passed with the local-argmax greedy path active:

| Gate | Result | Artifact |
| --- | ---: | --- |
| Arithmetic repeat, 32 persistent-engine calls | pass, all `42`, token-deterministic | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-arithmetic-alone-n64-r32-piecewise-local-argmax-20260517T011846Z.json` |
| Six-prompt semantic suite, 2 repeats | pass, prompt-scoped checks and deterministic hashes | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-extended-sixpack-n64-r2-piecewise-local-argmax-20260517T012756Z.json` |
| Raw145 exact hash, 64 generated tokens | pass, expected token hash `267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd` | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n64-piecewise-local-argmax-20260517T013050Z.json` |
| Raw145 exact hash, 256 generated tokens | pass, expected token hash `58f6e8251c7a0a17e8c441278b5861f7d5da914fa1823ecd10484b296f2d7537` | `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-raw145-n256-piecewise-local-argmax-20260517T013306Z.json` |

### Repeatable Benchmark Result

Configuration:

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Engine: vLLM `0.20.1-local`, XPU/Level Zero, TP4
- GPUs: 4x Intel Arc Pro B70 32GB
- Quantization: AutoRound W4A16 / INC INT4
- Prompt/decode: p512/n1536, batch 1, `max_model_len=2048`
- Runtime: llm-scaler INT4 MoE decode path, XPU piecewise graph, Triton attention, block size 256, prefix cache off, greedy `temperature=0`, no speculative decoding, no expert dropping, no power-limit increase

Repeat results:

| Run | Output tok/s | Total tok/s | Artifact |
| --- | ---: | ---: | --- |
| 20260517T012439Z | 60.3449 | 80.4599 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T012439Z.json` |
| 20260517T013627Z | 60.6495 | 80.8660 | `/home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n1536-20260517T013627Z.json` |
| Mean | 60.4972 | 80.6630 | two-run mean |

This is slightly below the earlier clean-weight-guard 61 tok/s submission, but it is stricter and more defensible: the known full-logits all-gather NaN boundary is no longer exercised for greedy decode, and the candidate passes repeat arithmetic, semantic, and raw exact-token canaries.

LocalMaxxing:

- Payload: `/home/steve/llm-optimizations-publish/data/localmaxxing-minimax-m27-autoround-local-argmax-quality-p512n1536-20260517.payload.json`
- Response: `/home/steve/llm-optimizations-publish/data/localmaxxing-responses/minimax-m27-autoround-local-argmax-quality-p512n1536-20260517.response.json`
- ID: `cmp940h1703tpo401scj5tftf`

### Remaining Limits

- This is not valid for temperature sampling, top-k/top-p sampling, logprobs, penalties, constrained decoding, or speculative decode unless those paths get equivalent correctness work.
- The full-vocab logits all-gather bug still needs a real fix. Local argmax is a quality-preserving bypass only for greedy decode.
- Next performance work should focus on replacing the tiny pair gather with a dedicated XPU collective/top-k reduction, and then on fusing the post-LM-head argmax into the model runner so the greedy decision stays entirely in the GPU-side decode path.
