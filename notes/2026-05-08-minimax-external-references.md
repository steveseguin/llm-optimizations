# 2026-05-08 MiniMax External References

## Sources Checked

- vLLM MiniMax-M2 recipe: https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
- Lasimeri MiniMax-M2.7 INT4 AutoRound: https://huggingface.co/Lasimeri/MiniMax-M2.7-int4-AutoRound
- vLLM Intel AutoRound quantization docs: https://docs.vllm.ai/en/v0.18.1/features/quantization/inc/
- Intel LLM Scaler: https://github.com/intel/llm-scaler
- DevRadar MiniMax M2.7 hardware comparison: https://devradar.dev/radar/minmax-m2-7-inference-benchmarks-rtx-hardware
- MJPansa REAP MiniMax AutoRound variant: https://huggingface.co/MJPansa/MiniMax-M2.7-REAP-172B-A10B-AutoRound-W4A16

## Takeaways For B70 Work

The official vLLM MiniMax-M2 recipe uses TP4 for 4 GPUs and explicitly says pure TP8 is not supported for the full model. For more than 4 GPUs, it points toward DP+EP or TP+EP, with TP4+EP recommended on H100-class systems. That means a future 8x B70 plan should not assume "just increase tensor parallelism"; expert parallel is likely the right abstraction if vLLM/XPU supports the required kernels.

The Lasimeri INT4 AutoRound checkpoint is W4A16 with group size 128 and MoE gate layers kept full precision. Its model card shows vLLM and SGLang examples, with TP8 as the published example for that checkpoint. For our current 4x B70 system, the first test should still be TP4; if it fails, capture the exact unsupported quant/MoE/XPU path before trying SGLang.

vLLM's Intel quantization docs confirm AutoRound is intended to produce INT2/3/4/8 and other formats and is integrated with Intel Neural Compressor. This makes the AutoRound download a valid path to try rather than a random quant variant.

Intel's LLM Scaler repo says it targets Arc Pro B60 and B70 and wraps standard frameworks such as vLLM, SGLang, ComfyUI, and Xinference. Treat it as a packaging and configuration reference first, not a new inference engine. Useful things to mine from it: XPU environment defaults, multi-GPU launch conventions, benchmark commands, and any CCL/Level Zero settings.

DevRadar's public MiniMax comparison gives rough external anchors: 4x RTX 4090 at `71.52 tok/s`, 4x RTX 5090 at `120.54 tok/s`, RTX PRO 6000 at `118.74 tok/s`, and DGX Spark at `24.41 tok/s`, all reportedly using UD-IQ3_XXS, 32k context, and 4096 max tokens. This is not directly comparable to our `17.335655 tok/s` UD-IQ4_XS p0/n64 decode test, but it supports the hypothesis that a better all-GPU/vLLM-style path should be able to beat the current GGUF RPC layer split.

The MJPansa REAP variant is an 86 GiB pruned AutoRound W4A16 checkpoint with a clear quality caveat. It is interesting as a capacity/performance fallback if full MiniMax remains too slow, but it is not the same target quality as unpruned MiniMax M2.7.

## Resulting Plan Adjustments

- Keep full MiniMax UD-IQ4_XS GGUF as the reproducible baseline.
- Test Lasimeri INT4 AutoRound in vLLM/XPU TP4 as soon as the USB download completes.
- If the AutoRound path fails, inspect whether the failure is quant kernel, custom MiniMax modeling code, MoE/expert parallel, CCL, or XPU memory management.
- Review Intel LLM Scaler configs before inventing more launch flags for vLLM/XPU.
- For future 8x B70 work, prioritize EP/DP+EP investigation over pure TP8.
