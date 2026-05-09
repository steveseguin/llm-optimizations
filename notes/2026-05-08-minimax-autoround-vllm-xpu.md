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

FP16 baseline check:

```text
p512/n128, TP4, pidfd, P2P=1, dtype=float16, no llm-scaler custom path:
100.832219 total tok/s, 20.17 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T011343Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T011343Z.json
LocalMaxxing: cmoxnvmna00gmml01eqdyl428
```

Interpretation: neutral/slightly positive versus the BF16 hybrid run. vLLM logs `Casting torch.bfloat16 to torch.float16`, so this is an activation dtype change and should be recorded separately from the BF16 baseline. It does not materially change the MiniMax ceiling, but it is a valid reproducibility point.

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

## Speculative Decode Checks

MiniMax M2.7 advertises MTP in its config:

```text
model_type=minimax_m2
use_mtp=True
num_mtp_modules=3
mtp_transformer_layers=1
```

The local AutoRound checkpoint does not include the extra MTP layer tensors that would be needed to run native MTP from this model:

```bash
jq -r '.weight_map | keys[]' /mnt/corsair-external/llm-models/minimax-m2.7-int4-autoround/model.safetensors.index.json | rg '^model\.layers\.(62|63|64)\.' | wc -l
# 0
```

This vLLM tree also has MiniMax code to skip speculative-layer weights when loading the base model, but no MiniMax MTP draft-model adapter. Interpretation: do not spend more time on native MiniMax MTP with this AutoRound checkpoint. It needs a checkpoint that actually carries the MTP weights plus a vLLM adapter, or a separate draft model.

N-gram speculation was tested through vLLM's `--speculative-config` path:

```text
p64/n16, TP4, hybrid B70 MoE config, method=ngram, num_speculative_tokens=4:
11.287082 total tok/s, 2.26 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260509T000544Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260509T000544Z.json

p64/n16, TP4, hybrid B70 MoE config, method=ngram_gpu, num_speculative_tokens=4:
15.728492 total tok/s, 3.15 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260509T001359Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p64n16-20260509T001359Z.json
```

Interpretation: negative for the current random single-session throughput harness. The non-speculative p64 path was around `13.6` output tok/s and the current promoted p512/n128 high is `20.11` output tok/s; both n-gram variants are far slower. CPU n-gram also disables async scheduling in vLLM, and GPU n-gram keeps async scheduling but still loses badly. These were not submitted to LocalMaxxing.

## llm-scaler INT4 MoE Kernel Path

`llm-scaler` is still worth pursuing because it contains custom ESIMD INT4 MoE kernels for small INT4 MoE routing on Intel GPUs:

```text
/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm
tests/test_moe_int4_kernel.py has "122B-A10B-TP4":
hidden_size=3072, intermediate_size=256, num_experts=256, top_k=8
```

That sample is MiniMax-like but not exact. The actual AutoRound checkpoint is:

```text
hidden_size=3072
intermediate_size=1536
num_local_experts=256
num_experts_per_tok=8
```

Under TP4, the local expert intermediate size is `384`, so the exact routed MoE microbench shape is `H=3072, D=384, E=256, top_k=8`.

The full extension and the MoE-only extension initially failed under oneAPI 2026 because PyTorch's older bundled SYCL headers do not define `__DPCPP_SYCL_EXTERNAL_LIBC`, while oneAPI 2026's `sycl/stl_wrappers/cmath` expects it. A minimal compile probe proved this workaround:

```text
-D__DPCPP_SYCL_EXTERNAL_LIBC=__DPCPP_SYCL_EXTERNAL
```

Local patch applied:

```text
/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/setup_moe_int4_only.py
```

oneAPI 2026 could build a MoE-only `.so`, but it either linked to `libsycl.so.9` and crashed on first XPU launch or linked to the PyTorch venv's `libsycl.so.8` and crashed at import. Installing the side-by-side oneAPI `2025.3.2` compiler fixed that ABI mismatch:

```text
icpx: Intel(R) oneAPI DPC++/C++ Compiler 2025.3.2
extension NEEDED: libsycl.so.8
RUNPATH: /home/steve/.venvs/vllm-xpu/lib
```

Smoke tests after the 2025.3.2 rebuild:

```text
moe_int4_ops import: pass
router tiny launch: pass
MiniMax-like full_int4 smoke bs=1,4,16: pass
```

Exact MiniMax TP4 routed MoE microbench:

```text
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/llm-scaler-minimax-routed-moe-vs-vllm-20260509T004707Z.log

bs=1:  vLLM fused_experts 355.7 us, llm-scaler N-major 91.1 us
bs=2:  vLLM fused_experts 338.9 us, llm-scaler N-major 158.7 us
bs=4:  vLLM fused_experts 397.0 us, llm-scaler N-major 305.2 us
bs=8:  vLLM fused_experts 743.2 us, llm-scaler N-major 380.2 us
bs=16: vLLM fused_experts 1052.8 us, llm-scaler N-major 509.1 us
bs=32: vLLM fused_experts 2437.4 us, llm-scaler N-major 726.7 us
bs=64: vLLM fused_experts 3132.9 us, llm-scaler N-major 1033.8 us
```

Interpretation: strong microbench positive. The exact routed MoE path is 1.3x to 3.9x faster than vLLM's current W4A16 `fused_experts` microbench for the MiniMax TP4 local shape.

Experimental vLLM integration:

```text
patches/vllm-minimax-llm-scaler-moe-experimental-20260509.patch
patches/llm-scaler-moe-int4-only-oneapi-sycl-compat-20260509.patch
```

The integration is opt-in with `VLLM_XPU_USE_LLM_SCALER_MOE=1`. It signs the uint4 expert weights in place with nibble-wise `xor 0x88`, avoiding a duplicate copy of the 62-layer expert weights. It is restricted to FP16 scales/activations because the llm-scaler tiny decode kernel asserts `torch::kHalf`.

Full model results:

```text
p512/n128, TP4, dtype=float16, VLLM_XPU_USE_LLM_SCALER_MOE=1:
61.374542 total tok/s, 12.27 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T010605Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T010605Z.json

p512/n128, TP4, dtype=float16, VLLM_XPU_USE_LLM_SCALER_MOE=0:
100.832219 total tok/s, 20.17 output tok/s
log: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T011343Z.log
json: /home/steve/bench-results/minimax-m2.7-autoround-vllm/vllm-minimax-m27-autoround-tp4-p512n128-20260509T011343Z.json
```

Interpretation: full-model negative as currently integrated. The low-level MoE kernel is promising, but the Python-level routed integration loses the end-to-end benchmark badly, likely because prefill routes through Python-side route/gather work and because the custom path cannot be used only for decode without changing the weight format. Do not enable `VLLM_XPU_USE_LLM_SCALER_MOE=1` for normal runs yet.

Next useful llm-scaler work:

- Add BF16 support to the tiny decode kernels or explicitly benchmark FP16 quality before relying on FP16.
- Add an unsigned-uint4 variant so vLLM can keep its current weight format and use the custom path only for decode, while preserving the faster existing prefill path.
- Move the exact N-major routed path into a proper C++/SYCL monolithic op to eliminate Python route/gather overhead for prompt sizes.
- Re-test full p512/n128 only after one of those changes lands.

## Open Items

- Continue submitting useful AutoRound records. Current best is `cmox94fsm0095ml01tjeb20rr` with the hybrid B70 MoE config.
- Keep `MAX_BATCHED_TOKENS=1024` for p512/n128; `512` is a large regression with the current hybrid MoE config.
- Keep `CCL_ZE_IPC_EXCHANGE=pidfd` as the current vLLM/XPU default; sockets is slower in the p64 smoke and earlier p512 run.
- Keep `CCL_TOPO_P2P_ACCESS=1`; `0` was slightly slower in the p64 smoke. Treat `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` as neutral/diagnostic because it improved p512 output only from `19.85` to `19.89` tok/s while overriding topology validation.
- Treat n-gram and GPU n-gram speculative decode as negative for the current MiniMax random single-session harness.
- Treat native MiniMax MTP as blocked for this checkpoint because no MTP layer tensors are present.
- Continue the llm-scaler ESIMD INT4 MoE path, but do not enable the current Python-level vLLM integration for production runs. The exact MoE microbench is strongly positive, while the full model path regresses to `12.27` output tok/s.
- Treat `--enable-expert-parallel` as negative/blocked for TP4 single-session MiniMax on B70. Untuned EP p64/n16 reached only `3.75` output tok/s, and the tuned EP config OOMed during model initialization.
- Treat `VLLM_XPU_ENABLE_XPU_GRAPH=1` as negative for TP4 MiniMax AutoRound until vLLM can capture communication ops or split capture around collectives.
- Treat `--enforce-eager` as negative for this path unless the compiled path starts failing; it regressed p64/n16 to `56.113901` total tok/s and `11.22` output tok/s.
- Treat `--compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` as blocked on XPU because this build lacks `torch.ops._C.minimax_allreduce_rms_qk`; implement or port that fused op before retesting for speed.
- Retune larger MiniMax MoE prompt sizes only if the microbench shows a stronger gain than default. The first hybrid config is a small positive but not a path to 30 tok/s by itself.
- Consider moving the AutoRound model to NVMe if iteration time, not decode speed, becomes the limiting factor.
