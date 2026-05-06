# 2026-05-06 FP8 vLLM post-reboot validation

## Summary

After reboot, the first standalone XCCL checks segfaulted at `dist.barrier()`.
The root cause was runtime library ordering: sourcing oneAPI `setvars.sh` before
torch/vLLM put `/opt/intel/oneapi` libraries ahead of the vLLM venv libraries.
With `/home/steve/.venvs/vllm-xpu-managed/lib` first in `LD_LIBRARY_PATH` and no
oneAPI `setvars.sh` for torch/vLLM, XCCL recovered.

Patch applied locally:

- `/home/steve/bench-vllm-qwen36-fp8.sh` now prepends `$VENV/lib` to
  `LD_LIBRARY_PATH`.
- The wrapper also defaults `CCL_ATL_TRANSPORT=ofi`.

## XCCL gate

Clean environment:

```bash
source /home/steve/.venvs/vllm-xpu-managed/bin/activate
export LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu-managed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export VLLM_TARGET_DEVICE=xpu
export CCL_ATL_TRANSPORT=ofi
```

Result:

- 2-rank barrier/allreduce probe: pass.
- Full 2-rank allreduce sweep: pass.
- 256 MiB payload: `41.75 GB/s`.
- Log: `/home/steve/bench-results/qwen36-fp8-vllm/xccl-standalone-2rank-venvlib-post-reboot-20260506T072239Z.log`

Failed mixed-library logs:

- `/home/steve/bench-results/qwen36-fp8-vllm/xccl-standalone-2rank-post-reboot-20260506T071737Z.log`
- `/home/steve/bench-results/qwen36-fp8-vllm/xccl-probe-barrier-2rank-20260506T072048Z.log`

## PP2 x TP2 fair 512/512 result

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`
- engine: vLLM/XPU `0.20.1`
- GPUs: 4x Arc Pro B70
- topology: pipeline parallel `2`, tensor parallel `2`
- quantization: `compressed-tensors` FP8
- prompt/output: `512/512`
- speculative decode: off
- `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=1024`

Result:

- latencies: `18.831783489`, `18.411271027`, `18.654142821` seconds
- average latency: `18.63239911233298` seconds
- output throughput: `27.479016 tok/s`
- total throughput: `54.958033 tok/s`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out512-bs1-20260506T072309Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out512-bs1-20260506T072309Z.log`

Decision: stable but not competitive. PP2 x TP2 needs speculative decode fixed
or a different scheduling layout before it is a useful single-session speed path.

## TP4 n-gram new best

Configuration:

- model: `/home/steve/models/qwen3.6-27b-fp8-vrfai`
- engine: vLLM/XPU `0.20.1`
- GPUs: 4x Arc Pro B70
- topology: tensor parallel `4`, pipeline parallel `1`
- quantization: `compressed-tensors` FP8
- attention: XPU FlashAttention2
- prompt/output: `512/512`
- speculative decode: n-gram, `num_speculative_tokens=4`, lookup min/max `2/4`
- `GPU_MEM_UTIL=0.80`, `MAX_MODEL_LEN=1024`

Result:

- latencies: `10.693160829`, `11.773327080`, `8.512563063` seconds
- average latency: `10.326350323998971` seconds
- output throughput: `49.581893 tok/s`
- total throughput: `99.163787 tok/s`
- JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T072633Z.json`
- log: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260506T072633Z.log`
- LocalMaxxing: `cmotql1v60013qy01016jcs7r`

Speculative acceptance varied heavily by iteration:

- iteration 1: mean acceptance length `2.89`, draft acceptance `48.0%`
- iteration 2: mean acceptance length `1.55`, draft acceptance `14.4%`
- iteration 3: mean acceptance length `3.25`, draft acceptance `57.4%`

Decision: this is the current validated FP8 best. It is a speed/quantization
tradeoff versus Q4_0 GGUF, but it does not lower quality via INT4 AutoRound.

## Next

- Keep torch/vLLM commands out of oneAPI `setvars.sh` shells unless
  `$VENV/lib` is explicitly first in `LD_LIBRARY_PATH`.
- Fix PP2+n-gram negative scheduled-token handling before retesting 2x2
  speculative decode.
- Consider a real 2-session layout on four cards: run two independent TP2 FP8
  sessions for aggregate throughput, while continuing single-session work on TP4.
