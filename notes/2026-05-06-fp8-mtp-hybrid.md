# 2026-05-06 FP8 MTP Hybrid Follow-Up

## Context

Goal: test Qwen3.6 27B MTP/speculative decode on the four B70 system without changing model quality for final outputs.

The fast static FP8 compressed-tensors directory (`/home/steve/models/qwen3.6-27b-fp8-vrfai`) has no `mtp.*` keys in `model.safetensors`. Earlier MTP runs against that directory resolved `Qwen3_5MTP`, but the drafter did not have real MTP weights and should not be treated as a valid MTP result.

The dynamic FP8 directory (`/home/steve/models/qwen3.6-27b-fp8-hf`) contains `mtp.safetensors` with 22 `mtp.*` tensors, including FP8 projection weights and `weight_scale_inv` tensors.

## Hybrid Setup

Created:

`/home/steve/models/qwen3.6-27b-fp8-vrfai-mtp-hybrid`

This is a symlinked model directory:

- all static compressed-tensors files symlink to `/home/steve/models/qwen3.6-27b-fp8-vrfai`;
- `mtp.safetensors` symlinks to `/home/steve/models/qwen3.6-27b-fp8-hf/mtp.safetensors`.

The loader sees this as a 2-shard checkpoint with `33.90 GiB` total size, confirming both the static main model and MTP shard are discovered.

## vLLM Loader Patch

Patch:

`/home/steve/llm-optimization-artifacts/patches/vllm-qwen35-mtp-loader-original-name-20260506.patch`

Problem fixed: `Qwen3_5MultiTokenPredictor.load_weights()` mutated `name` while iterating packed mapping candidates. If a candidate mapping did not exist in `params_dict`, later candidates matched the already-mutated name and produced bogus packed names such as `qkqkv_proj` and `gate_gate_up_proj`.

Patch behavior: keep `loaded_name` as the immutable checkpoint tensor name for each packed mapping candidate.

Validated with `py_compile`.

## Results

All runs used TP4/PP1 on the four B70s, input 32, output 8, max model len 1024.

- Dynamic FP8 with real MTP, eager: `8.330130855 s`, `0.960369 tok/s`.
- Static+MTP hybrid before loader patch, eager: `1.079748554 s`, `7.409132 tok/s`.
- Static+MTP hybrid after loader patch, eager: `1.023203455 s`, `7.818582 tok/s`.
- Static+MTP hybrid after loader patch, compiled/async: `13.752427987 s`, `0.581715 tok/s`.

For comparison, non-spec static FP8 short smoke from earlier was much faster at about `25 tok/s` for the same 32/8 shape, and validated static FP8 512/512 runs remain around `42-50 tok/s`.

## Remaining Issue

After the loader mutation patch, vLLM still skips real packed scale tensors:

- `layers.0.mlp.down_proj.weight_scale_inv`
- `layers.0.mlp.gate_up_proj.weight_scale_inv`
- `layers.0.self_attn.qkv_proj.weight_scale_inv`
- `layers.0.self_attn.o_proj.weight_scale_inv`

These are no longer bogus names; the compressed-tensors MTP parameter dict does not expose matching packed scale parameters for the hybrid MTP path.

## Decision

Do not submit these MTP results to LocalMaxxing. They are useful internal negative/diagnostic findings, but they are not clean leaderboard data because the hybrid compressed-tensors MTP scale loading is incomplete and performance is below validated non-spec baselines.

Next useful MTP work:

- inspect compressed-tensors/Fp8LinearMethod parameter creation for Qwen3_5MTP packed projection scales;
- prefer a clean static compressed-tensors model that includes MTP weights and matching quant metadata;
- revisit longer MTP benchmarks only after packed scale loading is clean.
