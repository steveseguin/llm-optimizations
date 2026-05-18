# 2026-05-18 MiniMax M2.7 MBT boundary

## Scope

This note records the strict quality-gated `max_num_batched_tokens` boundary for the current MiniMax M2.7 AutoRound 4x B70 recipe.

The goal was to see whether larger prefill graph ranges could improve the already-promoted p512/n1536 decode result without sacrificing output quality.

## Baseline recipe

- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Local path: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Dtype: FP16 activations, AutoRound INT4 W4A16 weights
- Attention: default XPU FlashAttention v2
- Graph mode: XPU graph, PIECEWISE
- MoE path: llm-scaler INT4 MoE decode with work-sharing enabled
- Shape: p512/n1536, ctx2048, batch 1, block size 256
- Quality gate: raw145 exact n64/n256 token hashes, semantic suite, 16-repeat arithmetic canary, extended sixpack

Key env:

```bash
VLLM_XPU_ENABLE_XPU_GRAPH=1
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
VLLM_XPU_USE_LLM_SCALER_MOE=1
VLLM_XPU_USE_LLM_SCALER_MOE_WS=1
VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT=1
VLLM_MINIMAX_QK_NORM_RESTORE_WEIGHT_MIN_TOKENS=2
CCL_TOPO_P2P_ACCESS=1
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
ZE_AFFINITY_MASK=0,1,2,3
```

Compile config:

```json
{"use_inductor_graph_partition":true,"compile_sizes":[1],"cudagraph_mode":"PIECEWISE"}
```

## Results

| MBT | Quality result | Mean output tok/s | Mean total tok/s | Interpretation |
| --- | --- | ---: | ---: | --- |
| 512 | strict pass | 80.602755 | 107.470340 | Current promoted LocalMaxxing result. |
| 768 | strict pass | 80.876005 | 107.834674 | Valid but only +0.34% output over MBT512, too small to republish as a meaningful win. |
| 832 | strict pass | 77.795833 | 103.727777 | Valid but slower than MBT512/768. |
| 896 | failed semantic suite | n/a | n/a | Exact raw145 n64/n256 passed, then semantic repeat corrupted into NUL/control output. |
| 1024 | failed raw145 n64 | n/a | n/a | Immediate NUL/control-token corruption before benchmarking. |

## MBT768 detail

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt768-strict-20260518-strict-tp4-ctx2048-mbt768-bs256-20260518T060634Z-summary.json`
- Repeats:
  - 80.553960 output tok/s, 107.405281 total tok/s
  - 81.198050 output tok/s, 108.264067 total tok/s
- Quality: all strict gates passed.

This is a valid result, but the gain over the current promoted MBT512 run is within normal run-to-run noise and not worth a new LocalMaxxing submission.

## MBT832 detail

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt832-strict-20260518-strict-tp4-ctx2048-mbt832-bs256-20260518T063952Z-summary.json`
- Repeats:
  - 78.051999 output tok/s, 104.069332 total tok/s
  - 77.539666 output tok/s, 103.386222 total tok/s
- Quality: all strict gates passed.

This proves `MAX_BATCHED_TOKENS=832` is not corrupt on the current stack, but it slows the p512/n1536 decode path.

## MBT896 failure

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt896-strict-20260518-strict-tp4-ctx2048-mbt896-bs256-20260518T062839Z-summary.json`
- Failed JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt896-strict-20260518-strict-tp4-ctx2048-mbt896-bs256-20260518T062839Z-quality/semantic-suite-n64-r2.json`
- Failure reasons:
  - `nondeterministic lstrip_text`
  - `degenerate or corrupt generated output`
  - `prompt-scoped required substring missing`
  - `prompt-scoped required regex missing or invalid`
- Corruption observed: `nul_token_count=256`, `control_char_output=true`, `degenerate_output=true`.

Important: the raw145 exact n64/n256 gates passed before this failure. That means exact single-prompt canaries are necessary but not sufficient for this graph boundary.

## MBT1024 failure

- Summary: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt1024-strict-20260518-strict-tp4-ctx2048-mbt1024-bs256-20260518T055959Z-summary.json`
- Failed JSON: `/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-moe-ws-flash-piecewise-mbt1024-strict-20260518-strict-tp4-ctx2048-mbt1024-bs256-20260518T055959Z-quality/raw145-n64-exact.json`
- Failure reasons:
  - `combined token hash mismatch`
  - `degenerate or corrupt generated output`
- Corruption observed: `distinct_generated_token_count=1`, `first_distinct_generated_tokens=[0]`, `nul_token_count=64`, `control_char_output=true`.

## Compiler note

At MBT768 and MBT832, the run emitted intermittent `ocloc` or IGC floating point exception messages while compiling `triton_red_fused__to_copy_mm_t_8`, but execution continued and strict quality passed.

This compiler warning is not by itself a failed candidate, but it is suspicious because higher MBT values later produced deterministic-looking graph execution with corrupted output. Treat it as a driver/compiler risk marker when testing larger graph ranges.

## Decision

Keep `MAX_BATCHED_TOKENS=512` as the promoted public setting. `MBT768` is valid but not materially better, `MBT832` is valid but slower, and `MBT896+` is unsafe on this driver/compiler/runtime stack.

No LocalMaxxing submission was made for these boundary runs because none produced a meaningful quality-safe improvement over the existing accepted result `cmpasdq5v007nmn019elaut3s`.

## Next steps

- Do not spend more time raising MBT for this p512/n1536 decode objective until the compiler/runtime changes.
- If prefill becomes the focus, benchmark MBT separately with prompt-heavy shapes and the same strict quality gate.
- Move optimization effort back to decode-critical work: hidden-state collective boundaries, MoE/projection epilogue fusion, and reducing CPU/framework callbacks without changing logits.
