# MiniMax UR Runtime And DFlash Screens, 2026-05-13

Goal: look for quality-preserving paths toward `60+` output tok/s on MiniMax
M2.7 AutoRound W4A16 without changing target logits. These tests kept the same
TP4 target model, FP16 activations, llm-scaler u4 MoE decode path, and static
single-token AOT graph unless noted.

## UR / Level Zero Runtime Knobs

| Test | Prompt/Output | Output tok/s | Total tok/s | Outcome |
| --- | ---: | ---: | ---: | --- |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` | 512/512 | 45.487 | 90.974 | negative |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS=2` | 512/512 | n/a | n/a | hung after AOT load |
| `UR_L0_DEVICE_SCOPE_EVENTS=2` | 512/512 | n/a | n/a | hung during init |
| `CCL_ATL_TRANSPORT=mpi` | 512/512 | n/a | n/a | hung during init |

`UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` completed and reused the current AOT
graph, but it was slightly slower than the accepted async/static p512/n512
screen. It also lowered reported GPU KV-cache capacity from `16,832` to
`16,064` tokens in that run, so there is no reason to repeat it at p512/n1536.

`UR_L0_USE_IMMEDIATE_COMMANDLISTS=2` loaded the AOT graph and allocated the
normal `16,832` KV-cache tokens, but then produced repeated Intel IGC/`ocloc`
internal compiler errors followed by the vLLM shared-memory broadcast timeout.

`UR_L0_DEVICE_SCOPE_EVENTS=2` and forcing oneCCL ATL back to `mpi` both stalled
before model-weight/AOT progress reached the normal warm-cache point. The
working recipe should keep the default UR event behavior and the wrapper's
`CCL_ATL_TRANSPORT=ofi` setting.

## DFlash Drafter Smoke

I downloaded the current MiniMax-M2.7 DFlash drafter:

- Hugging Face: `MirecX/MiniMax-M2.7-L3H5-DFlash`
- local path: `/mnt/fast-ai/llm-models/minimax-m2.7-l3h5-dflash`
- local max-512 smoke copy:
  `/mnt/fast-ai/llm-models/minimax-m2.7-l3h5-dflash-max512`

The upstream card is useful because it is a MiniMax-specific target-verified
DFlash route, but it is not expected to be a speed win as published: the card's
reported acceptance is below the break-even point for its reference TP4 run.
That matches local behavior.

Screens:

| Test | Prompt/Output | Outcome |
| --- | ---: | --- |
| original drafter config | 64/32 | stalled before generation; draft resolved `max_model_len=196608` |
| local max-512 config copy | 64/32 | still stalled before generation |

The max-512 config copy did fix the oversized draft model length in vLLM's
startup log, but it did not make DFlash runnable on this XPU path. Keep DFlash
on the roadmap only as a target-verified speculation path after the XPU/vLLM
initialization issue is understood; it is not a current `60 tok/s` candidate.

## Decision

No new LocalMaxxing submission. These are valid learnings, but not promoted
benchmark results.

The next MiniMax speed work should stay on source-level execution: reduce or
fuse the explicit allreduce/wait boundaries around residual/RMS/MoE/projection
work, or find a graph scheduling path that avoids additional waits without
changing Q/K variance allreduce, expert routing, or target verification.

## Logs

- immediate command lists mode 1:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ur-immediate1-staticcompile-p512n512-20260513T015504Z`
- immediate command lists mode 2:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ur-immediate2-staticcompile-p512n512-20260513T015725Z`
- device-scope events:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ur-devscope2-staticcompile-p512n512-20260513T020124Z`
- oneCCL MPI ATL:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-ccl-mpi-staticcompile-p512n512-20260513T020627Z`
- DFlash original:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dflash-l3h5-smoke-p64n32-20260513T021112Z`
- DFlash max-512 local copy:
  `/mnt/fast-ai/bench-results/minimax-m2.7-autoround-vllm-dflash-l3h5-max512-smoke-p64n32-20260513T021532Z`
