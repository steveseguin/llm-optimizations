# MiniMax Attention Post-Reduce RMS XPU Helper Quality Fail

Date: 2026-05-19

## Goal

Try a narrow attention-side helper that preserves the promoted collective order:
keep the existing `RowParallelLinear` attention output allreduce, then replace
only the post-reduce residual-add + RMSNorm step with a small SYCL helper behind
`VLLM_MINIMAX_ATTN_POST_REDUCE_RMS_XPU=1`.

This was intentionally different from the rejected AR+RMS candidates. It did
not move the allreduce, did not use delayed rank-0 residual ordering, and did
not change sampling.

## Build And Microcheck

The helper was added to the existing `minimax_ar_fused_rms_xpu` experiment as
`torch.ops.minimax_ar_fused_rms_xpu.add_rms(input, residual, weight, eps)`.

Rebuild used the oneAPI 2025.3 compiler to stay compatible with the active
PyTorch XPU runtime:

```text
source /home/steve/.venvs/vllm-xpu/bin/activate
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
export CC=/opt/intel/oneapi/compiler/2025.3/bin/icx
export CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx
export TORCH_XPU_ARCH_LIST=bmg
MAX_JOBS=2 python setup.py build_ext --inplace
```

A direct XPU microcheck against the intended FP32-add/RMS formula was bit-exact:

```text
max_out_diff: 0.0
max_res_diff: 0.0
```

## Model Result

Label:

```text
minimax-attn-post-reduce-rms-xpu-20260519
```

Summary:

```text
/home/steve/bench-results/minimax-m2.7-strict-candidates/minimax-minimax-attn-post-reduce-rms-xpu-20260519-strict-tp4-ctx2048-mbt512-bs256-20260519T124229Z-summary.json
```

Quality gate:

- `raw145-n64-exact`: failed exact combined token hash

Observed facts:

```text
expected token hash: 267cbf30208d84929ee79284ac695467f7e80597bf8694130e1e1f8b180eb5bd
observed token hash: cd6c4f7e39506f6038de1425e1a40c0512200de409c9cfee1f681d5b2be069f7
observed text hash:  c1062da8a35a873907c613f51a29f66cbf5c18a70d073730a250bc1cf4ae1cd9
failed n64 output speed: 9.52 tok/s
compile time: 146.33 s
reported KV cache: 9472 tokens
```

No p512/n1536 benchmark repeats were run because the first exact canary failed.

## Decision

Reject and do not submit to LocalMaxxing.

The isolated helper matched its local formula, but the integrated model output
changed immediately and the graph was much slower. This path is therefore both
not quality-preserving and not a viable speed path. The active vLLM source and
installed venv hook were removed after recording the result.

## Artifacts

- Data: `data/minimax-m27-attn-post-reduce-rms-xpu-quality-fail-20260519.json`
- Patch tried: `patches/minimax-attn-post-reduce-rms-xpu-quality-fail-20260519.md`
- Related earlier rejection: `notes/2026-05-19-minimax-ar-rms-ordered-negative.md`
