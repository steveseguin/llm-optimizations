# 2026-05-07 FP8 TP4 vs PP2xTP2 Refresh

## Context

After reboot and the Q4_0 guard-fix validation, I refreshed the static FP8 path using `vrfai/Qwen3.6-27B-FP8` with the local vLLM/XPU tree at `c51df4300` plus the existing local XPU patches. The goal was to re-check the proposed 2x2 layout and compare it against the known TP4 speed path at the same 512 prompt / 512 output shape.

All runs used `max_model_len=32768`, `QUANTIZATION=compressed-tensors`, `KV_CACHE_DTYPE=auto`, XPU FlashAttention2, `CCL_ATL_TRANSPORT=ofi`, no power-limit changes, and the reusable `scripts/bench-vllm-qwen36-fp8.sh` wrapper.

## Results

| Layout | Extra setting | Speculative | tok/s out | tok/s total | Avg latency |
| --- | --- | --- | ---: | ---: | ---: |
| PP2xTP2 | default CCL topology | no | 27.722318 | 55.444635 | 18.468874 s |
| TP4/PP1 | default CCL topology | no | 45.864956 | 91.729913 | 11.163207 s |
| TP4/PP1 | default CCL topology | n-gram 4, lookup 2..4 | 48.082178 | 96.164356 | 10.648436 s |
| TP4/PP1 | `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | no | 46.386137 | 92.772274 | 11.037781 s |
| TP4/PP1 | `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | n-gram 4, lookup 2..4 | 44.438715 | 88.877430 | 11.521485 s |

## Observations

- TP4 remains the speed layout for batch-1 Qwen3.6 27B FP8 on four B70s.
- PP2xTP2 is valuable as a capacity/layout validation: it fits a 32K context and reported about 1.15M GPU KV-cache tokens, but pipeline overhead makes it much slower for a single session.
- The oneCCL topology override is a small no-spec win, but it is not safe to generalize. With n-gram speculation, draft acceptance collapsed and the run regressed below no-spec TP4.
- vLLM logs show XPU graph disabled for TP/PP multi-GPU runs because XPU graph capture does not support communication ops when world size is greater than one.
- The previous submitted FP8 best, `49.581893 tok/s` with TP4 n-gram and default topology recognition, remains the headline FP8 result.

## LocalMaxxing

Submitted the PP2xTP2 capacity-focused negative result:

- ID: `cmout3vhy00m6ld01162ujv21`
- Model: `vrfai/Qwen3.6-27B-FP8`
- Engine: `vllm` `0.20.1`
- Quantization: `FP8 static compressed-tensors`
- Metrics: `27.722317582905 tok/s` output, `55.44463516581 tok/s` total

## Artifacts

- PP2xTP2 JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp2-pp2-in512-out512-bs1-20260507T010724Z.json`
- TP4 no-spec JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260507T011158Z.json`
- TP4 n-gram JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260507T011540Z.json`
- TP4 no-spec CCL override JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260507T012024Z.json`
- TP4 n-gram CCL override JSON: `/home/steve/bench-results/qwen36-fp8-vllm/vllm-qwen36-fp8-compressed-tensors-tp4-pp1-in512-out512-bs1-20260507T012303Z.json`
