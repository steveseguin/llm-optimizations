# MiniMax Compiled NaN Boundary

Goal: keep MiniMax M2.7 AutoRound W4A16 optimization honest before treating the
fast compiled TP4 path as a valid leaderboard/runtime result.

## Summary

The default compiled vLLM path is not quality-valid right now. It emits token
`0` / NUL characters, and a finite-value trace shows the failure is already
present in the sampled final hidden state before the LM head and sampler.

The enforced-eager path remains quality-valid on the same model, prompt, TP4
setup, llm-scaler INT4 MoE path, and delayed-attention allreduce source patch.
That means the checkpoint, tokenizer, weight load, TP4 setup, and eager model
math are not inherently broken.

## Key Artifacts

- Compiled finite trace:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-finite-trace-20260514T023956Z.{json,log,trace.jsonl}`
- Compiled without attention delayed allreduce:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-noattndelay-finite-trace-20260514T024850Z.{json,log,trace.jsonl}`
- Compiled with a fresh AOT cache:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-freshcache-finite-trace-20260514T025123Z.{json,log,trace.jsonl}`
- Enforced-eager quality pass:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/eager-enforced-quality-postgate-20260514T025943Z.{json,log}`
- Compiled quality failure with tightened gate:
  `/home/steve/bench-results/minimax-m2.7-integrity-gate/compiled-quality-postgate-20260514T030205Z.{json,log}`

## Findings

- Default compiled mode emits 16 NUL tokens on the 16-token smoke and now fails
  the quality gate.
- The one-token compiled trace showed warmup tensors were finite, but the actual
  request had `gpu_model_runner.sample_hidden_states` as all NaN:
  `shape=[1,3072]`, `nan=3072`, `finite=0`.
- The LM-head local logits and gathered logits were consequently all NaN.
- Disabling `VLLM_MINIMAX_M2_ATTN_DELAY_ALLREDUCE` did not fix the compiled
  NaNs, so the issue is not isolated to that optimization.
- Forcing a fresh `VLLM_CACHE_ROOT` rebuilt the AOT payload and still produced
  all-NaN request hidden states, so this is not stale cache contamination.
- `compilation_mode=none` produced the first token correctly (`" Paris"`) but
  then collapsed to NUL tokens. It is not a valid fallback.
- `--enforce-eager` produced normal text:
  `Paris. The capital of Germany is Berlin...` with no control characters.

## Harness Changes

The MiniMax quality gate now treats generated NUL/control-character output as a
failure:

- `control_nonspace_text_chars`
- `nul_token_count`
- `control_char_output`

This closes the gap where a result with one printable token followed by NUL
tokens could pass only because the token set had more than one distinct value.

## Runtime Diagnostic Patch

The active vLLM venv and `/home/steve/src/vllm` source tree now include an
opt-in diagnostic helper:

- `vllm/utils/xpu_finite_trace.py`
- `VLLM_XPU_FINITE_TRACE=1`
- `VLLM_XPU_FINITE_TRACE_FILE=/path/to/trace.jsonl`
- `VLLM_XPU_FINITE_TRACE_LIMIT=4`

The trace is intentionally sync-heavy and should only be used for tiny
correctness probes. It records compact JSONL tensor finite/NaN/Inf stats around
sample hidden states and logits.

## Current Decision

Do not submit compiled MiniMax AutoRound speed numbers to LocalMaxxing until the
compiled path passes the quality gate. Existing fast compiled MiniMax results
should be treated as performance leads, not promoted quality-preserving results.

The safe near-term baseline is enforced eager. It is slower, but it preserves
quality and lets us compare future fixes honestly.

## Next Work

- Find the first NaN-producing boundary inside the compiled model forward without
  turning the whole path into eager execution.
- Inspect the generated Inductor/AOT code around the first request path versus
  warmup path. Warmup is finite; the scheduled request is not.
- Try smaller compile scopes or explicit graph breaks around attention/KV/cache
  update boundaries to isolate the bad compiled section.
- Once a compiled quality pass is recovered, rerun the throughput matrix and only
  then promote results to LocalMaxxing.
