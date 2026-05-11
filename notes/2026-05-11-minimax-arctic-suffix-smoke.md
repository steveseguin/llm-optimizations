# MiniMax Arctic Suffix Speculative Smoke, 2026-05-11

## Goal

Test whether vLLM 0.20.1's native `method: "suffix"` speculative path can be
made runnable on the XPU MiniMax AutoRound stack without installing the full
Arctic Inference plugin.

## Build Outcome

The full Arctic Inference 0.1.2 package is not a clean fit for this host:

- its optional vLLM extra pins `vllm==0.11.0`, while the active runtime is the
  local `0.20.1` XPU stack;
- the broader plugin package imports CUDA-specific pieces;
- vLLM 0.20.1's suffix proposer only needs
  `arctic_inference.suffix_decoding`.

I built a minimal suffix-only package at:

- `/home/steve/src/arctic-suffix-only`

The reproducible build wrapper is:

```bash
/home/steve/llm-optimizations-publish/scripts/build-arctic-suffix-only.sh
```

The wrapper intentionally fetches or reuses the Arctic source tarball directly
instead of running `pip download`, because pip build isolation tries to resolve
the full Arctic build dependency chain, including a CUDA/Torch stack that is not
needed for suffix decoding.

Use it by prepending the suffix package before the llm-scaler kernels:

```bash
export PYTHONPATH=/home/steve/src/arctic-suffix-only:/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python:$PYTHONPATH
```

Direct smoke passed: `vllm.utils.import_utils.has_arctic_inference()` returned
true and `SuffixDecodingCache.speculate()` returned draft tokens.

## vLLM Smoke

Model:

- `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`

Suffix run:

```bash
OUTDIR=/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-suffix-smoke \
MODEL=/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround \
TP=4 INPUT_LEN=64 OUTPUT_LEN=16 NUM_PROMPTS=1 \
MAX_MODEL_LEN=256 MAX_BATCHED_TOKENS=128 MAX_NUM_SEQS=1 DTYPE=float16 \
USE_LLM_SCALER_MOE=1 \
VLLM_CACHE_ROOT=/mnt/fast-ai/vllm-cache/minimax-suffix-smoke-$(date -u +%Y%m%dT%H%M%SZ) \
LLM_SCALER_KERNELS=/home/steve/src/arctic-suffix-only:/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python \
EXTRA_ARGS='--speculative-config {"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":16,"suffix_decoding_max_cached_requests":128,"suffix_decoding_min_token_prob":0.0}' \
timeout 720s /home/steve/llm-optimizations-publish/scripts/bench-vllm-minimax-autoround-xpu.sh
```

Result:

- total: `18.322240` tok/s
- output-equivalent: `3.664` tok/s
- elapsed: `4.366278` s
- log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-suffix-smoke/vllm-minimax-m27-autoround-tp4-p64n16-20260511T004151Z.log`
- json:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-suffix-smoke/vllm-minimax-m27-autoround-tp4-p64n16-20260511T004151Z.json`

Matching no-spec run:

- total: `18.192808` tok/s
- output-equivalent: `3.638` tok/s
- elapsed: `4.397342` s
- log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-nospec-p64n16/vllm-minimax-m27-autoround-tp4-p64n16-20260511T004532Z.log`
- json:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-nospec-p64n16/vllm-minimax-m27-autoround-tp4-p64n16-20260511T004532Z.json`

vLLM warning of note:

```text
Async scheduling not supported with suffix-based speculative decoding and will be disabled.
```

## Interpretation

This is technically useful but not a speed result. The suffix proposer now runs
on the B70/XPU vLLM stack, but on vLLM's random benchmark it is essentially tied
with no-spec and far below the normal p512/n1536 reference. Suffix speculation
also disables async scheduling, so it needs high accepted-token rates from a
repetitive or cache-friendly prompt distribution before it can pay for itself.

Do not submit this screen to LocalMaxxing. Revisit it only with:

- a repetition-heavy prompt set;
- multi-request cache reuse where suffix decoding has a chance to match prior
  responses;
- explicit acceptance behavior if vLLM exposes it for suffix runs.
