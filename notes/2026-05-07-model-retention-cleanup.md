# Model Retention Cleanup

Date: 2026-05-07

## Keep Set

The active model set is now focused on the targets that matter for the B70 optimization work:

- Qwen3.6 27B Q4_0 GGUF: `/home/steve/models/qwen3.6-27b-q4_0-gguf`
- Qwen3.6 27B Q4_0 flat fused beta/alpha GGUF: `/home/steve/models/qwen3.6-27b-q4_0-fused-ba-gguf`
- Qwen3.6 27B FP8 static/vLLM path: `/home/steve/models/qwen3.6-27b-fp8-vrfai`
- Qwen3.6 27B FP8 HF/official-style path: `/home/steve/models/qwen3.6-27b-fp8-hf`
- MiniMax M2.7 UD-IQ4_XS GGUF: `/home/steve/models/minimax-m2.7-ud-iq4_xs-gguf/UD-IQ4_XS`

I kept both Qwen3.6 FP8 variants because FP8 remains a primary target and the two trees cover different runtime experiments. The tiny `qwen3.6-27b-fp8-vrfai-mtp-hybrid` directory was left in place because it is only 16 KiB.

## Removed

These were removed because they are not in the current target set and can be re-downloaded if needed:

- `/home/steve/models/qwen3.5-4b-gguf`
- `/home/steve/models/qwen3.5-27b-fp8-gguf`
- `/home/steve/models/qwen3.6-27b-q8_0-gguf`
- `/home/steve/models/hf/Lorbus-Qwen3.6-27B-int4-AutoRound`

Disk result:

- Before: `/` was `98%` full with about `11G` free.
- After: `/` is `83%` full with about `75G` free.

## dmesg Notes

`sudo dmesg --ctime --level=err,warn` showed expected workstation noise plus several B70/xe warnings worth remembering:

- `xe ... GT0: Schedule disable failed to respond` on multiple B70 PCI functions.
- `xe ... VM worker error: -16` on one B70 PCI function.
- Several `pm_runtime_work hogged CPU` warnings.
- NIC PCIe bandwidth warnings for the `i40e` device, not the B70s.

No new cleanup action was taken based on these logs. The xe warnings are consistent with the earlier driver/runtime instability notes and should stay part of the driver-quality watchlist.
