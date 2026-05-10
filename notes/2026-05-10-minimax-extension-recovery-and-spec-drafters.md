# MiniMax Extension Recovery And Spec Drafter Screens, 2026-05-10

## Current Target

MiniMax M2.7 AutoRound INT4 is now the primary 4x B70 target. Since the
AutoRound path is materially faster than the original GGUF capacity path, the
working targets are higher:

- quality-preserving non-speculative target: `60+` output tok/s at p512/n1536;
- target-verified speculative stretch: `75+` output tok/s;
- keep reporting total/prefill-inclusive tok/s as well as output/decode tok/s;
- no expert dropping, no skipped Q/K RMS variance allreduce, no power changes.

Qwen 27B/35B remains useful as a dense-model comparison track, but MiniMax
M2.7 AutoRound is the main multi-GPU software optimization path.

## llm-scaler MoE Extension Recovery

The active `moe_int4_ops` extension had been replaced by a broader experimental
build that crashed in SYCL registration. The useful recovery path was to rebuild
only the MoE INT4 extension against the oneAPI 2025.3 compiler/runtime shape
that matches the vLLM XPU virtualenv, then copy that MoE-only `.so` over the
broken active extension.

Recovered source/build root:

```text
/home/steve/src/llm-scaler-u4-e0only/vllm/custom-esimd-kernels-vllm
```

Active extension restored:

```text
/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/moe_int4_ops.cpython-312-x86_64-linux-gnu.so
```

Broken experimental extension was kept for forensics:

```text
/home/steve/src/llm-scaler/vllm/custom-esimd-kernels-vllm/python/custom_esimd_kernels_vllm/moe_int4_ops.cpython-312-x86_64-linux-gnu.so.broken-20260510T201409Z
```

Important build/import rule:

```bash
source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
source /home/steve/.venvs/vllm-xpu/bin/activate
export LD_LIBRARY_PATH=/home/steve/.venvs/vllm-xpu/lib:${LD_LIBRARY_PATH:-}
cd /home/steve/src/llm-scaler-u4-e0only/vllm/custom-esimd-kernels-vllm
rm -rf build/temp.linux-x86_64-cpython-312 build/lib.linux-x86_64-cpython-312
python setup_moe_int4_only.py build_ext --inplace
```

Do not build/import this extension with oneAPI 2026 first in `LD_LIBRARY_PATH`
on the current stack. That path reproduced the `__sycl_register_lib` /
`ProgramManager::addImage` crash.

## Recovered Clean Floor

Clean p512/n1536 run after extension recovery:

- model: `/mnt/fast-ai/llm-models/minimax-m2.7-int4-autoround`
- engine: vLLM/XPU TP4, FP16, llm-scaler INT4 MoE path
- shape: p512/n1536, `max_model_len=2048`, `max_num_batched_tokens=1024`
- output tok/s: `37.724645`
- total tok/s: `50.299527`
- elapsed: `40.716089 s`
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-recovered-clean-p512n1536/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T201653Z.log`
- JSON: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-recovered-clean-p512n1536/vllm-minimax-m27-autoround-tp4-p512n1536-20260510T201653Z.json`

This is a healthy recovered floor and slightly above the earlier accepted public
anchor (`37.552538` output tok/s / `50.070051` total tok/s), but it is not a new
LocalMaxxing submission because it does not materially change the public result.

## Speculative Drafter Screens

### M2.5 EAGLE3 Draft

Downloaded draft:

```text
/mnt/fast-ai/llm-models/spec-drafts/MiniMax-M2.5-Eagle3
```

Screen A, target TP4 plus draft TP1:

- config: `method=eagle3`, `num_speculative_tokens=4`, `draft_tensor_parallel_size=1`
- shape: p64/n128
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-eagle3-m25-p64n128/vllm-minimax-m27-autoround-tp4-p64n128-20260510T202353Z.log`
- result: no JSON; loaded and compiled, then stalled at processed prompts with repeated `No available shared memory broadcast block found in 60 seconds`.

Screen B, target TP4 plus draft TP4:

- config: `method=eagle3`, `num_speculative_tokens=2`, `draft_tensor_parallel_size=4`
- shape: p64/n32
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-eagle3-m25-tp4-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T203021Z.log`
- result: no JSON; stalled during early distributed initialization/load before
  any token generation.

Conclusion: the M2.5 EAGLE3 draft is compatible enough to initialize, but the
current vLLM XPU speculative execution path is blocked by multiprocess/shared
memory scheduling before it can produce a valid throughput result.

### M2.7 DFlash Draft

Downloaded MiniMax-specific DFlash draft:

```text
/mnt/fast-ai/llm-models/spec-drafts/MiniMax-M2.7-L3H5-DFlash
```

The drafter config uses `DFlashDraftModel` with target auxiliary layers:

```text
2, 16, 30, 43, 57
```

Screen:

- config: `method=dflash`, `num_speculative_tokens=4`, `draft_tensor_parallel_size=1`
- shape: p64/n32
- log: `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dflash-m27-p64n32/vllm-minimax-m27-autoround-tp4-p64n32-20260510T203735Z.log`
- result: no JSON; target and drafter loaded, auxiliary layers were detected,
  target and draft graphs compiled, then generation stalled at processed prompts
  with the same shared-memory broadcast-block warning.

Conclusion: DFlash is the right class of quality-preserving speculation for
MiniMax, but it is currently blocked in the vLLM XPU scheduling/broadcast path
before acceptance rate or throughput can be measured.

## Next Work

Speculation stays on the roadmap, but the immediate path back to throughput is
non-speculative TP/MoE work:

1. keep the clean MoE extension active and avoid oneAPI 2026-built extension
   artifacts until the SYCL registration issue is isolated;
2. continue source-level screens around hidden-state allreduce plus
   residual/RMSNorm and MoE epilogue fusion;
3. only revisit DFlash/EAGLE3 after a small XPU scheduler/shared-memory fix or
   a vLLM upstream change makes the speculative path complete a p64 smoke run;
4. submit to LocalMaxxing only once a result is valid, reproducible, and useful
   with both total and output tok/s recorded.
