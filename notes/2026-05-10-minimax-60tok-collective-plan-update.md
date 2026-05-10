# MiniMax 60 Tok/S Collective Plan Update, 2026-05-10

## Target

Primary model remains `Lasimeri/MiniMax-M2.7-int4-AutoRound` on 4x Intel Arc
Pro B70 with vLLM/XPU TP4 and the local llm-scaler u4 decode MoE path.

Updated aspiration targets:

- non-speculative, quality-preserving: `60+` output tok/s at p512/n1536;
- speculative, target-verified only: `75+` output tok/s;
- always report total/prefill-inclusive tok/s beside decode/output tok/s for
  LocalMaxxing-worthy runs.

Quality guardrails are unchanged: do not skip Q/K TP variance allreduce, do not
drop experts, do not promote root-residual/deferred-reduction shortcuts without
logit checks, and do not count a lower-quality quantization as a clean speedup.

## Latest External Signals

- Intel llm-scaler currently lists `intel/llm-scaler-vllm:0.14.0-b8.2.1`
  as the latest vLLM image and explicitly calls out Arc Pro B70 support,
  CCL P2P/USM, INT4/FP8 serving, and TP/PP/DP support.
  Reference: `https://github.com/intel/llm-scaler/blob/main/Releases.md`
  and `https://github.com/intel/llm-scaler`.
- vLLM's latest speculative-decoding docs expose `draft_model`, `ngram`,
  `suffix`, `mtp`, `eagle3`, and `dflash` through `--speculative-config`.
  Reference: `https://docs.vllm.ai/en/latest/features/speculative_decoding/`.
- vLLM's P-EAGLE writeup is useful for the stretch target because it moves
  EAGLE draft token generation from sequential draft passes to a parallel
  single-pass drafter. No ready MiniMax P-EAGLE draft was found locally.
  Reference: `https://vllm.ai/blog/p-eagle`.
- Public MiniMax M2.7 hardware numbers keep the target ambitious but plausible:
  UD-IQ3_XXS llama.cpp results cite `71.52` tok/s on 4x RTX 4090,
  `120.54` tok/s on 4x RTX 5090, `118.74` tok/s on RTX PRO 6000, and
  `24.41` tok/s on DGX Spark. Backend and quantization differ, so these are
  aspiration references rather than direct apples-to-apples baselines.
  Reference: `https://devradar.dev/radar/minmax-m2-7-inference-benchmarks-rtx-hardware`.

## Current Local Evidence

Current quality-conservative long anchor:

| Shape | Total tok/s | Output tok/s | Notes |
| --- | ---: | ---: | --- |
| p512/n1536 | `50.070051` | `37.552538` | Accepted LocalMaxxing `cmozow03v005wlo01q81bnspx`; Q/K TP variance allreduce enabled; no speculation; no expert dropping. |
| p512/n1536 post-reboot refresh | `49.583865` | `37.19` | Valid repeat; no new LocalMaxxing submission because it duplicates the anchor. |

Recent short screens:

| Screen | Shape | Total tok/s | Output tok/s | Decision |
| --- | --- | ---: | ---: | --- |
| plain baseline | p64/n128 | `50.418637` | `33.61` | short-run reference |
| DFlash no-async, 4 draft tokens | p64/n32 | `7.291369` | `2.43` | negative |
| DFlash no-async, 1 draft token | p64/n128 | `12.665482` | `8.44` | negative |
| `ngram_gpu`, 4 draft tokens | p64/n128 | `14.974028` | `9.98` | negative on random prompts |
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` | p64/n128 | `50.460186` | `33.64` | neutral |
| Q/K helper fusion pass | p64/n128 | `24.019450` | `16.01` | negative |
| XPU graph flag | p64/n128 | `22.982853` | `15.32` | negative; communication capture disabled |
| compile out-of-place allreduce | p64/n128 | `48.876167` | `32.58` | neutral/slightly negative |

The important timing result is from the opt-in allreduce/MoE timing hook:

- compiled synchronized timing at p64/n128 shows only visible MoE timing because
  compiled allreduces are hidden inside the AOT graph; MoE accounts for roughly
  `8.1 ms` per decode token in that distorted timing run;
- eager synchronized timing at p64/n32 exposes the collective count:
  `125` hidden-state `f16[(1,3072)]` allreduces per decode token, `62`
  Q/K scalar `f32[(1,2)]` allreduces per decode token, and `62` llm-scaler
  u4 MoE calls per decode token.

This means the path to `60+` is mostly a communication-boundary and graph
scheduling problem. The u4 MoE path is still relevant, but replacing only the
expert matvec cannot recover a 2x-3x gap.

## Next Implementation Order

1. Add finer call-site labels around hidden-state TP allreduces so the p64/n32
   eager timing can split the `125` hidden reductions into embedding,
   output-projection, MoE-output, and any final/vocab calls.
2. Target the largest repeated boundary first: hidden-state allreduce followed
   immediately by residual add/RMSNorm after `o_proj` or MoE output.
3. Keep Q/K RMS variance allreduce intact for now. Earlier Q/K helper/direct
   paths were correct or plausible in isolation but regressed compiled vLLM.
4. Revisit DFlash/EAGLE only after scheduler/runtime work; the current XPU
   implementation completes but is far slower than non-spec decode.
5. Promote only improvements that beat the p64/n128 short reference first, then
   rerun p512/n1536 and submit to LocalMaxxing with both total and output tok/s.

## Reproduction Artifacts

- benchmark wrapper:
  `scripts/bench-vllm-minimax-autoround-xpu.sh`
- timing hook patch:
  `patches/vllm-xpu-allreduce-moe-timing-20260510.patch`
- structured data:
  `data/minimax-m27-collective-plan-update-20260510.json`
