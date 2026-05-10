# MiniMax M2.7 Speculation, MTP, and DP4+EP Compile Follow-Up

Date: 2026-05-10

## Updated Target

MiniMax M2.7 AutoRound INT4 is now the main 4x B70 optimization target.

- Non-speculative target: repeatable `60+` output tok/s at p512/n1536.
- Speculative/MTP target: `75+` output tok/s only if target-model verification is preserved.
- Quality constraints remain strict: no expert dropping, no skipped Q/K TP variance allreduce, no root-residual ordering shortcut, and no unverified speculative result counted as a win.
- Power limits remain unchanged; this is software/runtime/compiler/kernel work only.

## Native MTP Status

The local AutoRound checkpoint advertises MTP fields:

```json
{"use_mtp":true,"num_mtp_modules":3,"mtp_transformer_layers":1,"num_hidden_layers":62}
```

But the safetensors index has no `model.layers.62+` weights. The highest target layer present is `61`, so there are no local MTP module weights to load.

The local vLLM speculative config has explicit MTP model mappings for DeepSeek, MiMo, GLM, Qwen3 Next, Qwen3.5, LongCat, Step3.5, HY, and others, but it does not map `minimax_m2` to a MiniMax MTP drafter. The local `MiniMaxM2ForCausalLM` implementation only exposes a helper to identify MTP layer weight names if such weights exist.

Decision: native MiniMax MTP is blocked for this AutoRound checkpoint. Do not count the config flag as evidence that usable MTP exists.

## Suffix Decoding Status

vLLM supports suffix decoding as a model-free speculative method and documents it as useful for repetitive agentic/code-editing workloads. It requires the `arctic_inference` package. The active vLLM/XPU venv does not have `arctic_inference` installed.

I inspected the `arctic-inference==0.1.1` package metadata. It is source-only on PyPI and its build-system requirements include `torch == 2.7.0`, `nanobind`, `cmake`, `ninja`, and `grpcio-tools`. A blind install into the active XPU vLLM environment would risk disturbing the working PyTorch/vLLM stack.

Decision: keep suffix decoding on the roadmap, but only test it from an isolated venv or source checkout. Do not install it directly into `/home/steve/.venvs/vllm-xpu` without pinning and verifying that it will not replace the current XPU Torch/vLLM packages.

References:

- vLLM suffix decoding docs: https://docs.vllm.ai/en/stable/features/speculative_decoding/suffix/
- Arctic Inference suffix decoding docs: https://arcticinference.readthedocs.io/en/latest/suffix-decoding.html

## DP4+EP Compiled Follow-Up

Earlier DP4+EP work proved that the local CCL-rank patch lets four data/expert-parallel ranks initialize and generate in eager mode, but compiled mode failed during Inductor profiling with an XPU OOM:

```text
Tried to allocate 1.15 GiB.
GPU total capacity: 31.89 GiB.
Free memory at failure: about 0.65-0.68 GiB.
PyTorch allocated: 30.81 GiB.
PyTorch reserved but unallocated: about 160 MiB.
```

I ran two follow-up probes:

1. Lowered `--gpu-memory-utilization` to `0.90`, disabled llm-scaler MoE, kept compiled mode.
2. Same as above, plus disabled Inductor `combo_kernels` and `benchmark_combo_kernel` via `--compilation-config`.

Both failed with the same 1.15 GiB Inductor autotuning allocation. Lower KV reservation and combo-kernel toggles do not free enough compile scratch because the model weights dominate the per-rank footprint.

Decision: DP4+EP compiled mode is closed for now on 32GB B70s with this AutoRound layout. Revisit only if one of these changes lands:

- lower per-rank model allocation, likely from a lighter quant or better EP weight layout;
- an Inductor/XPU compile path that avoids the 1.15 GiB profiling allocation;
- a server path that can use eager DP4+EP with a much faster XPU all-to-all/EP backend.

## Current Priority

The remaining high-value work remains TP4 graph/collective fusion:

1. Q/K variance allreduce plus RMS apply.
2. Hidden-state output-projection allreduce plus residual/RMSNorm.
3. MoE output allreduce plus downstream residual/RMS/projection epilogue work.
4. Attention/KV scheduling only where it removes a visible graph boundary or launch/fence.

The current AOT graph shows these boundaries once per layer, so a useful speedup needs fewer visible collectives/fences or a larger fused kernel around them. Flag-level routes are mostly exhausted.

## Artifacts

- DP4+EP low-util compiled log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-serve-dp4-ep-compiled-lowutil-noscaler/vllm-minimax-m27-autoround-serve-server-tp1-p16n8-np1-20260510T233752Z.log`
- DP4+EP no-combo compiled log:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-serve-dp4-ep-compiled-nocombo-noscaler/vllm-minimax-m27-autoround-serve-server-tp1-p16n8-np1-20260510T234120Z.log`
