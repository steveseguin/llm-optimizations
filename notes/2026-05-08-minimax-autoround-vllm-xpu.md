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

P2P toggle check:

```text
p64/n16, TP4, CCL_ZE_IPC_EXCHANGE=pidfd, CCL_TOPO_P2P_ACCESS=0:
62.410028 total tok/s, 12.48 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T172901Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T172901Z.json
```

Interpretation: keep `CCL_TOPO_P2P_ACCESS=1`. Setting it to `0` was slightly slower than the `pidfd` + P2P=1 p64 smoke (`68.171339` total tok/s, `13.63` output tok/s).

Topology-recognition diagnostic:

```text
p64/n16, TP4, pidfd, P2P=1, CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0:
71.182650 total tok/s, 14.24 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T173644Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T173644Z.json

p512/n128, TP4, pidfd, P2P=1, CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0:
99.459983 total tok/s, 19.89 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T174346Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T174346Z.json
```

Interpretation: neutral diagnostic. It suppresses the PCIe topology check and slightly improves the short smoke, but the p512/n128 result is only `+0.04` output tok/s over the accepted `19.85` result. Keep the conservative default without this override unless repeated longer runs prove a stable gain.

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

Eager-mode check:

```text
p64/n16, TP4, pidfd, P2P=1, --enforce-eager:
56.113901 total tok/s, 11.22 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T175150Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T175150Z.json
```

Interpretation: negative. Upstream Intel wNa16 docs recommend eager in some contexts, but for this MiniMax AutoRound TP4 path it disables torch.compile/CUDAGraphs and regresses versus the compiled `pidfd/P2P=1` p64 smoke (`68.171339` total, `13.63` output).

## B70 MoE Config Tune

The vLLM MoE tuning harness was CUDA-centric: Ray did not expose Intel XPUs as GPU resources, workers defaulted to `cuda`, and timing relied on CUDA graph capture. A local harness patch adds:

```text
patches/vllm-benchmark-moe-xpu-tune-harness-20260508.patch
```

The patch lets XPU users set `VLLM_MOE_BENCH_NUM_GPUS=4`, uses `current_platform.device_type`, falls back to synchronized eager timing for XPU, and prunes single-batch XPU decode shapes. The unpruned `M=1` tune had 1,920 configs and reached an hours-long ETA; the pruned decode tune cut that to 96 configs and completed in `57.93s`.

Generated B70 config:

```text
configs/vllm/minimax-m27-b70-int4-w4a16-moe-hybrid-20260508.json
```

The file is deliberately hybrid:

- key `1`: tuned B70 decode config, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=128`, `GROUP_SIZE_M=1`, `num_warps=4`, `num_stages=4`, `SPLIT_K=1`
- keys `64`, `256`, `512`: default vLLM prompt-size config, `BLOCK_SIZE_M=64`, `GROUP_SIZE_M=1`, `SPLIT_K=1`

The decode-only config was valid but slightly slower end-to-end on p64/n16 because vLLM reused key `1` for prompt shapes:

```text
p64/n16, TP4, pidfd, P2P=1, tuned key 1 only:
67.725172 total tok/s, 13.55 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T181416Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T181416Z.json
```

The hybrid config improved the real p512/n128 benchmark:

```text
p512/n128, TP4, pidfd, P2P=1, hybrid B70 MoE config:
100.538158 total tok/s, 20.11 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T182318Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T182318Z.json
LocalMaxxing: cmox94fsm0095ml01tjeb20rr
```

Interpretation: small positive. This moves the current MiniMax AutoRound high from `19.85` to `20.11` output tok/s and from `99.231127` to `100.538158` total tok/s. It is not enough to explain the full gap to the 30 tok/s target, but it proves that B70-specific MoE configs are active and can move end-to-end throughput.

## Expert Parallel Check

vLLM exposes `--enable-expert-parallel`, and it is functional on this stack. With TP4 it shards MiniMax experts to `64` local / `256` global experts per rank:

```text
p64/n16, TP4, pidfd, P2P=1, --enable-expert-parallel:
18.744070 total tok/s, 3.75 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T183643Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T183643Z.json
```

Interpretation: negative. EP greatly underperforms the TP4 non-EP path for batch-1 single-session decode. The likely bottleneck is all2all/expert-parallel scheduling overhead, not raw expert weight memory.

An EP-specific `E=64,N=1536` pruned MoE tune did find a much faster standalone decode kernel:

```text
default EP M=1 MoE kernel: ~723 us
tuned EP M=1 MoE kernel: ~277 us
config: configs/vllm/minimax-m27-b70-int4-w4a16-moe-ep-negative-20260508.json
```

However, the model-level run with that EP config failed during initialization:

```text
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 144.00 MiB.
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260508T184915Z.log
```

Interpretation: blocked/negative. Do not spend more full model-load cycles on EP for 4x B70 single-session MiniMax until the all2all cost and tuned-config memory behavior are understood.

## Chunked Prefill Knob

The current p512/n128 high uses `--max-num-batched-tokens 1024`. Reducing that to `512` with the same hybrid B70 MoE config regressed badly:

```text
p512/n128, TP4, pidfd, P2P=1, hybrid B70 MoE config, max_num_batched_tokens=512:
67.835347 total tok/s, 13.57 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T185133Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260508T185133Z.json
```

Interpretation: negative. Keep `MAX_BATCHED_TOKENS=1024` for the p512/n128 MiniMax AutoRound benchmark shape.

## Open Items

- Continue submitting useful AutoRound records. Current best is `cmox94fsm0095ml01tjeb20rr` with the hybrid B70 MoE config.
- Keep `MAX_BATCHED_TOKENS=1024` for p512/n128; `512` is a large regression with the current hybrid MoE config.
- Keep `CCL_ZE_IPC_EXCHANGE=pidfd` as the current vLLM/XPU default; sockets is slower in the p64 smoke and earlier p512 run.
- Keep `CCL_TOPO_P2P_ACCESS=1`; `0` was slightly slower in the p64 smoke. Treat `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` as neutral/diagnostic because it improved p512 output only from `19.85` to `19.89` tok/s while overriding topology validation.
- Treat `--enable-expert-parallel` as negative/blocked for TP4 single-session MiniMax on B70. Untuned EP p64/n16 reached only `3.75` output tok/s, and the tuned EP config OOMed during model initialization.
- Treat `VLLM_XPU_ENABLE_XPU_GRAPH=1` as negative for TP4 MiniMax AutoRound until vLLM can capture communication ops or split capture around collectives.
- Treat `--enforce-eager` as negative for this path unless the compiled path starts failing; it regressed p64/n16 to `56.113901` total tok/s and `11.22` output tok/s.
- Treat `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` as blocked on XPU because this build lacks `torch.ops._C.minimax_allreduce_rms_qk`; implement or port that fused op before retesting for speed.
- Retune larger MiniMax MoE prompt sizes only if the microbench shows a stronger gain than default. The first hybrid config is a small positive but not a path to 30 tok/s by itself.
- Consider moving the AutoRound model to NVMe if iteration time, not decode speed, becomes the limiting factor.
