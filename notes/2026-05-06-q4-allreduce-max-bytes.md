# Q4_0 fused allreduce max-byte probe

Date: 2026-05-06

## Change

Added `GGML_META_FUSE_ALLREDUCE_MAX_BYTES` as an opt-in ceiling for the existing meta backend fused allreduce paths. The default remains `64 KiB`, preserving current behavior.

The knob applies to:

- `PARTIAL -> ADD`
- `PARTIAL -> RESHAPE -> ADD`
- `PARTIAL -> GET_ROWS`
- `PARTIAL -> RESHAPE`

Patch scope: `/home/steve/src/llama.cpp-q4-b70/ggml/src/ggml-backend-meta.cpp`.

## Validation

Build target:

```bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cmake --build /home/steve/src/llama.cpp-q4-b70/build-sycl-2026-bmg-g31 --target llama-bench -j 8
```

Small graph probe, `p32/n1`, TP3, Qwen3.6 27B Q4_0:

- default ceiling: prompt projection residuals did not fuse; decode projection residuals mostly fused.
- `GGML_META_FUSE_ALLREDUCE_MAX_BYTES=1048576`: prompt projection residuals fused too.

Counts with raised ceiling:

- prompt `linear_attn_out`: `48/48` fused, all via reshape
- prompt `attn_output`: `15/16` fused
- prompt `ffn_out`: `63/63` fused
- decode `linear_attn_out`: `48/48` fused, all via reshape
- decode `attn_output`: `15/16` fused
- decode `ffn_out`: `65/65` fused

The recurring miss is the final-layer `attn_output -> GET_ROWS` branch.

## Performance

Quick TP3 prompt-only comparison at 512 tokens:

- default: `195.625197 tok/s`
- `GGML_META_FUSE_ALLREDUCE_MAX_BYTES=16777216`: `196.233518 tok/s`

Full TP3 `p512/n512`, `-ub 128`, with raised ceiling:

- prompt: `188.183896 tok/s`
- decode: `49.346817 tok/s`

Prior best RMS_NORM+MUL TP3 run:

- prompt: `206.347938 tok/s`
- decode: `49.366188 tok/s`

## Interpretation

The knob works and can expose larger prompt/prefill allreduce+ADD fusion, but it did not improve the validated 512/512 speed path. It is useful as a diagnostic or long-prefill experiment, not a new default and not LocalMaxxing-worthy.

The decode path is already covered for almost every regular projection residual. The next real Q4_0 speed attempt should move below this meta boundary: either a dedicated output-projection partial GEMV plus allreduce/residual epilogue, or a final-layer `GET_ROWS`+residual variant if that path becomes measurable.

## Files

- default graph probe: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/meta-allreduce-stats4-tp3-rmsmul-p32n1-20260506T211134Z.log`
- raised graph probe: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/meta-allreduce-max1m-stats4-tp3-rmsmul-p32n1-20260506T212110Z.log`
- prompt default: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/default-tp3-rmsmul-p512n1-20260506T212233Z.jsonl`
- prompt raised: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/max16m-tp3-rmsmul-p512n1-20260506T212338Z.jsonl`
- full raised: `/home/steve/bench-results/qwen36-q4_0-gguf/next-fusion-probes-20260506/max16m-tp3-rmsmul-p512n512-r2-20260506T212502Z.jsonl`
