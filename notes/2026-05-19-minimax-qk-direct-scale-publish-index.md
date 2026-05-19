# MiniMax Q/K Direct Scale Publish Index

Date: 2026-05-19

This index records the connector-side GitHub publish for the current MiniMax clean-path follow-up.

## Current Clean Result

- Label: `minimax-qk-direct-inplace-scale-20260519b`
- Model: `Lasimeri/MiniMax-M2.7-int4-AutoRound`
- Hardware: 4x Intel Arc Pro B70 32GB
- Engine: vLLM `0.20.1-local`, XPU TP4
- Shape: p512/n1536, ctx2048, batch 1
- Result: `88.501953` output tok/s, `118.002604` total tok/s
- Samples: `88.739272`, `88.302351`, `88.821529`, `88.144660`
- Quality: raw145 n64/n256 exact hashes, semantic suite, arithmetic repeat n64 r16, extended sixpack all passed
- LocalMaxxing: `cmpc8cmqm0060pc016g5l5ukh`, `APPROVED`

## What Changed

`VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1` routes decode-sized FP32 Q/K variance tensors through `vllm.all_reduce_inplace` directly at the MiniMax call site, then scales `qk_var` in-place. Wider tensors keep the existing generic TP allreduce path.

This is a small clean-path win over the prior Q/K-helper result (`88.313105` output tok/s) and alias-correct tiny-FP32 in-place baseline (`88.103866` output tok/s). It remains slightly below the warning-prone skip-clone speed headline (`88.748424` output tok/s), but avoids that PyTorch alias warning.

## Rejected Screen

`VLLM_MINIMAX_QK_RMS_XPU_HELPER_MAX_TOKENS=512` was quality-safe but slower:

- Mean: `87.974187` output tok/s, `117.298916` total tok/s
- Decision: reject, do not submit to LocalMaxxing
- Lesson: keep the Q/K helper decode-sized (`MAX_TOKENS=4`) for now; widening it into prompt/profile token ranges changes the compiled schedule and loses speed.

## Published Artifacts

- `notes/2026-05-19-minimax-qk-direct-inplace-scale.md`
- `notes/2026-05-19-minimax-qk-helper-max512-negative.md`
- `data/minimax-m27-qk-direct-inplace-scale-20260519.json`
- `data/minimax-m27-qk-helper-max512-negative-20260519.json`
- `data/localmaxxing-minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.payload.json`
- `data/localmaxxing-responses/minimax-m27-autoround-qk-direct-inplace-scale-p512n1536-20260519.response.json`
- `patches/minimax-qk-direct-inplace-scale-20260519.patch`

## Next Work

Use this clean result as the next baseline for quality-gated attempts. The most plausible next avenues are residual/allreduce boundary fusion, MoE/projection epilogue fusion, and a custom fused Q/K variance allreduce+apply path that removes more framework overhead without changing hashes.
