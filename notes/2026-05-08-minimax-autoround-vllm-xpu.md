# 2026-05-08 MiniMax AutoRound vLLM/XPU Bring-Up

## Goal

Evaluate `Lasimeri/MiniMax-M2.7-int4-AutoRound` as an alternative to the current UD-IQ4_XS GGUF path. The target is not just fitting the model; it needs to beat the GGUF RPC+SYCL layer-mode ceiling and eventually move toward the 30 tok/s MiniMax goal without changing power limits.

## Model And Environment

- Model path: `/mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround`
- vLLM environment: `/home/steve/.venvs/vllm-xpu`
- vLLM source checkout: `/home/steve/src/vllm`, revision around `c51df4300`
- Hardware: 4x Intel Arc Pro B70, TP4 over XPU/XCCL
- Filesystem: USB drive mounted with NTFS3; model load takes about 5m48s because the 112.43 GiB checkpoint is larger than available RAM and vLLM disables auto-prefetch.

## Quantized MoE Fit Patch

Unpatched vLLM 0.20.1 detects the AutoRound config as `quantization=inc`, but the XPU W4A16 INC path only returned a quant method for dense `LinearBase` and `ParallelLMHead` layers. MiniMax `FusedMoE` layers therefore fell back to the XPU unquantized MoE backend and OOMed at load:

```text
Using XPU Unquantized MoE backend
XPU out of memory. Tried to allocate 1.12 GiB.
```

The local experimental patch routes `FusedMoE` through `MoeWNA16Config` using the AutoRound bit/group/sym settings:

```text
patches/vllm-inc-xpu-autoround-fusedmoe-wna16-20260508.patch
```

After the patch, the model loads successfully:

```text
Model loading took 28.11 GiB memory
```

This is the first useful AutoRound MiniMax result: the model is viable on 4x B70 if expert weights stay quantized.

## Upstream References

- Hugging Face model card: `Lasimeri/MiniMax-M2.7-int4-AutoRound` is W4A16, group size 128, with MoE `gate` layers kept full precision. The card's vLLM example is TP8 on NVIDIA-class hardware, so our TP4 B70 run is a platform adaptation rather than the documented target path. Source: https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound
- vLLM MiniMax recipe: official MiniMax M2.7 guidance recommends TP4 for four GPUs and notes that larger-than-four GPU setups should use DP/EP or TP/EP rather than pure TP8. It also recommends the `fuse_minimax_qk_norm` compile pass where available. Source: https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
- vLLM Intel quantization RFC: W4A16 linear support is merged, but W4A16 MoE on Intel GPU is still listed as planned. That matches the local failure mode and explains why the FusedMoE dispatch patch is needed. Source: https://github.com/vllm-project/vllm/issues/37979
- vLLM Intel quantization docs: current wNa16 deployment guidance for Intel GPU/CPU says to use `--enforce-eager`. This is now on the fallback list if compiled throughput continues to hit XPU/Triton/runtime issues. Source: https://docs.vllm.com.cn/en/latest/features/quantization/inc/

## Runtime Repairs

The active vLLM package was version-skewed against the source checkout. Targeted repairs copied matching source files into the venv:

- `vllm/v1/sample/thinking_budget_state.py`
- `vllm/v1/sample/metadata.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/request.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/v1/engine/__init__.py`

Failure progression:

- Before `thinking_budget_state.py`: import failed before model load.
- Before the `FusedMoE` quant patch: model load OOMed because XPU MoE stayed unquantized.
- Before `python3.12-dev`: Triton Intel launcher compile failed on missing `Python.h`.
- Before `v1/core/sched/output.py`: worker expected `NewRequestData.prompt_is_token_ids`.
- Before `v1/request.py` and engine input dataclasses: scheduler created `NewRequestData` from an older `Request` object without `prompt_is_token_ids`.

Python headers were also required for Triton Intel launcher compilation:

```text
python3.12-dev libpython3.12-dev
```

## Current Command Shape

Reusable wrapper:

```bash
scripts/bench-vllm-minimax-autoround-xpu.sh
```

Equivalent command shape:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3 \
ZE_AFFINITY_MASK=0,1,2,3 \
CCL_ATL_TRANSPORT=ofi \
CCL_ZE_IPC_EXCHANGE=sockets \
CCL_TOPO_P2P_ACCESS=1 \
VLLM_XPU_ENABLE_XPU_GRAPH=0 \
vllm bench throughput \
  --backend vllm \
  --model /mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround \
  --tokenizer /mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --max-model-len 512 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --dataset-name random \
  --random-input-len 64 \
  --random-output-len 16 \
  --random-range-ratio 0 \
  --num-prompts 1 \
  --disable-log-stats
```

## Status

The model now loads and generates through vLLM/XPU TP4.

First proof-of-life smoke:

```text
p64/n16, TP4, max_model_len=512:
46.059 total tok/s, 9.21 output tok/s from the progress meter
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-inc-xpu-moe-wna16-requestfix-tp4-p64n16-20260508T161843Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-inc-xpu-moe-wna16-requestfix-tp4-p64n16-20260508T161843Z.json
```

This is not a promoted benchmark; it is a tiny single-request smoke.

First p512/n128 measurement:

```text
p512/n128, TP4, max_model_len=2048:
67.259054 total tok/s, 13.45 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T162541Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T162541Z.json
```

Interpretation: AutoRound is now viable. This first p512 run was below the GGUF record (`17.693021` decode tok/s at p512/n128), but it was later superseded by the `pidfd` IPC result below. The log warns that vLLM is using the default MoE config because there is no B70-specific `E=256,N=384,dtype=int4_w4a16` config. That remains the next optimization target.

`pidfd` oneCCL IPC:

```text
p64/n16, TP4, max_model_len=512, CCL_ZE_IPC_EXCHANGE=pidfd:
68.171339 total tok/s, 13.63 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T171257Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T171257Z.json

p512/n128, TP4, max_model_len=2048, CCL_ZE_IPC_EXCHANGE=pidfd:
99.231127 total tok/s, 19.85 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T171955Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T171955Z.json
LocalMaxxing: cmox6tys30085ml0125gihg18
```

Interpretation: `pidfd` is the current vLLM/XPU MiniMax AutoRound default. It beats the earlier sockets/default p512 AutoRound run (`13.45` output tok/s) and the current GGUF p512 decode result (`17.693` output tok/s), while using the same AutoRound W4A16 weights and no power-limit changes. The wrapper default was changed to `CCL_IPC=pidfd`.

AMD-derived MoE config seed:

```text
Source seed:
/home/steve/src/vllm/vllm/model_executor/layers/fused_moe/configs/E=384,N=256,device_name=AMD_Instinct_MI355_OAM,dtype=int4_w4a16.json

B70 target filename:
/home/steve/bench-results/minimax-m2.7-autoround-vllm/tuned-configs-amd-int4-seed/E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json
```

The raw AMD seed was accepted by the config loader but failed at Triton compile time because `matrix_instr_nonkdim` is not recognized on this XPU/Triton path:

```text
RuntimeError: Worker failed with error ''Keyword argument matrix_instr_nonkdim was specified but unrecognised''
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T163648Z.log
```

After stripping `matrix_instr_nonkdim`, the same AMD-derived tile choices completed but regressed badly:

```text
p64/n16, TP4, max_model_len=512, stripped AMD-derived MoE config:
8.632747 total tok/s, 1.73 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T164639Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T164639Z.json
```

Interpretation: the external config mechanism works, but AMD Instinct W4A16 MoE tiling is a poor seed for B70. Do not promote this config or run larger benchmarks with it. A real B70 tuning pass is still needed.

XPU graph smoke:

```text
p64/n16, TP4, max_model_len=512, VLLM_XPU_ENABLE_XPU_GRAPH=1:
18.149238 total tok/s, 3.63 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T165550Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T165550Z.json
```

This is not promoted. vLLM explicitly disables graph capture on this TP4 path:

```text
XPU Graph doesn't support capture communication ops, disabling cudagraph_mode.
```

MiniMax QK-norm fusion smoke:

```text
EXTRA_ARGS='--compilation-config {"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'
```

The pass flag is accepted and vLLM logs `Enabled custom fusions: minimax_qk_norm`, but this build does not expose the fused Lamport op:

```text
hasattr(torch.ops._C, "minimax_allreduce_rms_qk") == False
```

The first run then failed with a pass-manager import guard issue:

```text
NameError: name 'MiniMaxQKNormPass' is not defined
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T170331Z.log
```

Patch snapshot retained:

```text
patches/vllm-minimax-qknorm-passmanager-xpu-guard-20260508.patch
```

Interpretation: QK-norm fusion is not currently a MiniMax/XPU speed path. The guard patch prevents a crash, but a useful optimization needs an XPU implementation of `minimax_allreduce_rms_qk` or an equivalent fused collective+RMS kernel.

## Open Items

- Submit useful AutoRound results as diagnostic records, while keeping the current GGUF path as the higher-performance MiniMax route.
- Keep `CCL_ZE_IPC_EXCHANGE=pidfd` as the current vLLM/XPU default; sockets is slower in the p64 smoke and earlier p512 run.
- Compare `CCL_TOPO_P2P_ACCESS=1` versus `0` and, only as a diagnostic, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`.
- Treat `VLLM_XPU_ENABLE_XPU_GRAPH=1` as negative for TP4 MiniMax AutoRound until vLLM can capture communication ops or split capture around collectives.
- Try `--enforce-eager` if the compiled path keeps failing; upstream Intel quantization docs currently recommend eager for wNa16.
- Treat `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` as blocked on XPU because this build lacks `torch.ops._C.minimax_allreduce_rms_qk`; implement or port that fused op before retesting for speed.
- Add or tune a B70 MoE config file for `E=256,N=384,device_name=Intel(R)_Graphics_[0xe223],dtype=int4_w4a16.json`; an AMD-derived seed regressed to `1.73` output tok/s on p64/n16.
- Consider moving the AutoRound model to NVMe if iteration time, not decode speed, becomes the limiting factor.
